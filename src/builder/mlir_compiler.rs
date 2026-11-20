#![cfg(feature = "mlir")]

use std::collections::HashMap;

use melior::{
    dialect::{
        arith::{self, CmpfPredicate, CmpiPredicate},
        cf, func,
    },
    ir::{
        attribute::{FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        block::BlockLike,
        operation::OperationLike,
        r#type::{FunctionType, IntegerType},
        region::RegionLike,
        Block, BlockRef, Location, Module, Region, Type, Value,
    },
    Context, ExecutionEngine,
};

use crate::builder::{
    paper::Paper,
    part::{
        BinaryOperationType, DecimalCalculateType, DecimalComparisonType, IntegerCalculateType,
        IntegerComparisonType, Part, Size, UnaryMathType,
    },
};


pub struct MlirCompiler<'c> {
    context: &'c Context,
    module: Module<'c>,
}

impl<'c> MlirCompiler<'c> {

    pub fn new(context: &'c Context) -> Self {
        let registry = melior::dialect::DialectRegistry::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let module = Module::new(Location::unknown(context));

        Self { context, module }
    }


    pub fn module(&self) -> &Module<'c> {
        &self.module
    }


    pub fn verify(&self) -> bool {
        self.module.as_operation().verify()
    }


    pub fn to_string(&self) -> String {
        self.module.as_operation().to_string()
    }


    pub fn compile_paper(&mut self, paper: &Paper) -> Result<(), String> {
        let location = Location::unknown(self.context);


        let has_int_return = paper.parts.iter().any(|p| matches!(p, Part::ReturnInteger));
        let return_types: Vec<Type<'c>> = if has_int_return {
            vec![IntegerType::new(self.context, 64).into()]
        } else {
            vec![]
        };

        let fn_type = FunctionType::new(self.context, &[], &return_types);
        let fn_attr = TypeAttribute::new(fn_type.into());
        let fn_name = StringAttribute::new(self.context, &paper.name);

        let region = Region::new();
        let entry_block = region.append_block(Block::new(&[]));

        let mut lowerer = MlirFunctionLowerer::new(self.context, location);
        lowerer.lower_paper_body(&region, entry_block, paper)?;

        let function = func::func(
            self.context,
            fn_name,
            fn_attr,
            region,
            &[],
            location,
        );

        self.module.body().append_operation(function);
        Ok(())
    }


    pub fn create_execution_engine(
        &self,
        optimization_level: usize,
        shared_library_paths: &[&str],
    ) -> ExecutionEngine {
        ExecutionEngine::new(&self.module, optimization_level, shared_library_paths, false, true)
    }


    pub fn execute(
        &self,
        entry_func: &str,
        arguments: &mut [*mut ()],
        optimization_level: usize,
        shared_library_paths: &[&str],
    ) -> Result<(), String> {
        let engine = self.create_execution_engine(optimization_level, shared_library_paths);
        unsafe {
            engine
                .invoke_packed(entry_func, arguments)
                .map_err(|e| format!("MLIR execution error: {:?}", e))
        }
    }
}


struct MlirFunctionLowerer<'c, 'a> {
    context: &'c Context,
    location: Location<'c>,
    stack: Vec<Value<'c, 'a>>,
    locals: HashMap<String, Value<'c, 'a>>,
    globals: HashMap<String, Value<'c, 'a>>,
}

impl<'c, 'a> MlirFunctionLowerer<'c, 'a> {
    fn new(context: &'c Context, location: Location<'c>) -> Self {
        Self {
            context,
            location,
            stack: Vec::new(),
            locals: HashMap::new(),
            globals: HashMap::new(),
        }
    }

    fn push_value(&mut self, val: Value<'c, 'a>) {
        self.stack.push(val);
    }

    fn pop_value(&mut self) -> Result<Value<'c, 'a>, String> {
        self.stack
            .pop()
            .ok_or_else(|| "Stack underflow: tried to pop an empty stack".to_string())
    }

    fn peek_value(&self) -> Result<Value<'c, 'a>, String> {
        self.stack
            .last()
            .copied()
            .ok_or_else(|| "Stack underflow: tried to peek an empty stack".to_string())
    }

    fn size_to_integer_type(&self, size: Size) -> Type<'c> {
        IntegerType::new(self.context, size as u32).into()
    }

    fn size_to_float_type(&self, size: Size) -> Result<Type<'c>, String> {
        match size {
            Size::S16 => Ok(Type::float16(self.context)),
            Size::S32 => Ok(Type::float32(self.context)),
            Size::S64 => Ok(Type::float64(self.context)),
            _ => Err(format!("Unsupported float size in MLIR lowering: {:?}", size)),
        }
    }


    fn emit_runtime_call(
        &mut self,
        block: BlockRef<'c, 'a>,
        symbol: &str,
        arg_count: usize,
        has_result: bool,
    ) -> Result<(), String> {
        let prev_len = self.stack.len();
        if prev_len < arg_count {
            return Err(format!(
                "Stack underflow: function '{}' requires {} args, but stack has {}",
                symbol, arg_count, prev_len
            ));
        }
        let args: Vec<Value<'c, 'a>> = self.stack.drain(prev_len - arg_count..).collect();
        let callee = FlatSymbolRefAttribute::new(self.context, symbol);
        let result_types: Vec<Type<'c>> = if has_result {
            vec![IntegerType::new(self.context, 64).into()]
        } else {
            vec![]
        };

        let call_op = func::call(self.context, callee, &args, &result_types, self.location);
        let op_ref = block.append_operation(call_op);
        if has_result && op_ref.result_count() > 0 {
            self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
        }
        Ok(())
    }

    fn lower_paper_body(
        &mut self,
        region: &'a Region<'c>,
        entry_block: BlockRef<'c, 'a>,
        paper: &Paper,
    ) -> Result<(), String> {
        let mut current_block = entry_block;
        let mut label_blocks: HashMap<String, BlockRef<'c, 'a>> = HashMap::new();


        for part in &paper.parts {
            if let Part::Label(label_name) = part {
                let block = region.append_block(Block::new(&[]));
                label_blocks.insert(label_name.clone(), block);
            }
        }

        for part in &paper.parts {
            match part {
                Part::Label(label_name) => {
                    let next_block = *label_blocks.get(label_name).unwrap();
                    if current_block.terminator().is_none() {
                        let br_op = cf::br(&next_block, &[], self.location);
                        current_block.append_operation(br_op);
                    }
                    current_block = next_block;
                }
                Part::Jump(target) => {
                    let target_block = label_blocks
                        .get(target)
                        .ok_or_else(|| format!("Jump target label '{}' not found", target))?;
                    let br_op = cf::br(target_block, &[], self.location);
                    current_block.append_operation(br_op);
                }
                Part::JumpIfFalse(target) => {
                    let target_block = *label_blocks
                        .get(target)
                        .ok_or_else(|| format!("Jump target label '{}' not found", target))?;
                    let fallthrough_block = region.append_block(Block::new(&[]));
                    let condition = self.pop_value()?;
                    let cond_br_op = cf::cond_br(
                        self.context,
                        condition,
                        &fallthrough_block,
                        &target_block,
                        &[],
                        &[],
                        self.location,
                    );
                    current_block.append_operation(cond_br_op);
                    current_block = fallthrough_block;
                }
                Part::JumpIfTrue(target) => {
                    let target_block = *label_blocks
                        .get(target)
                        .ok_or_else(|| format!("Jump target label '{}' not found", target))?;
                    let fallthrough_block = region.append_block(Block::new(&[]));
                    let condition = self.pop_value()?;
                    let cond_br_op = cf::cond_br(
                        self.context,
                        condition,
                        &target_block,
                        &fallthrough_block,
                        &[],
                        &[],
                        self.location,
                    );
                    current_block.append_operation(cond_br_op);
                    current_block = fallthrough_block;
                }
                Part::JumpIfNull(target) => {
                    let target_block = *label_blocks
                        .get(target)
                        .ok_or_else(|| format!("Jump target label '{}' not found", target))?;
                    let fallthrough_block = region.append_block(Block::new(&[]));
                    let val = self.pop_value()?;
                    let zero_attr = IntegerAttribute::new(val.r#type(), 0);
                    let zero_op = arith::constant(self.context, zero_attr.into(), self.location);
                    let zero_val = current_block.append_operation(zero_op).result(0).unwrap().into();

                    let cmp_op = arith::cmpi(self.context, CmpiPredicate::Eq, val, zero_val, self.location);
                    let is_null = current_block.append_operation(cmp_op).result(0).unwrap().into();

                    let cond_br_op = cf::cond_br(
                        self.context,
                        is_null,
                        &target_block,
                        &fallthrough_block,
                        &[],
                        &[],
                        self.location,
                    );
                    current_block.append_operation(cond_br_op);
                    current_block = fallthrough_block;
                }
                Part::JumpIfNonNull(target) => {
                    let target_block = *label_blocks
                        .get(target)
                        .ok_or_else(|| format!("Jump target label '{}' not found", target))?;
                    let fallthrough_block = region.append_block(Block::new(&[]));
                    let val = self.pop_value()?;
                    let zero_attr = IntegerAttribute::new(val.r#type(), 0);
                    let zero_op = arith::constant(self.context, zero_attr.into(), self.location);
                    let zero_val = current_block.append_operation(zero_op).result(0).unwrap().into();

                    let cmp_op = arith::cmpi(self.context, CmpiPredicate::Ne, val, zero_val, self.location);
                    let is_non_null = current_block.append_operation(cmp_op).result(0).unwrap().into();

                    let cond_br_op = cf::cond_br(
                        self.context,
                        is_non_null,
                        &target_block,
                        &fallthrough_block,
                        &[],
                        &[],
                        self.location,
                    );
                    current_block.append_operation(cond_br_op);
                    current_block = fallthrough_block;
                }
                Part::TableSwitch { default, cases } => {
                    let flag = self.pop_value()?;
                    let default_block = *label_blocks
                        .get(default)
                        .ok_or_else(|| format!("Default label '{}' not found", default))?;
                    let mut case_values = Vec::new();
                    let mut case_destinations = Vec::new();

                    for (val, target) in cases {
                        case_values.push(*val);
                        let target_block = *label_blocks
                            .get(target)
                            .ok_or_else(|| format!("Case label '{}' not found", target))?;
                        case_destinations.push((target_block, &[] as &[Value<'c, '_>]));
                    }

                    let switch_op = cf::switch(
                        self.context,
                        &case_values,
                        flag,
                        flag.r#type(),
                        (&default_block, &[]),
                        &case_destinations,
                        self.location,
                    )
                    .map_err(|e| e.to_string())?;

                    current_block.append_operation(switch_op);
                    current_block = region.append_block(Block::new(&[]));
                }
                _ => {
                    self.lower_part(current_block, part)?;
                }
            }
        }


        if current_block.terminator().is_none() {
            let ret_op = if let Some(val) = self.stack.pop() {
                func::r#return(&[val], self.location)
            } else {
                func::r#return(&[], self.location)
            };
            current_block.append_operation(ret_op);
        }

        Ok(())
    }

    fn lower_part(&mut self, block: BlockRef<'c, 'a>, part: &Part) -> Result<(), String> {
        let location = self.location;
        match part {

            Part::Pop => {
                self.pop_value()?;
            }
            Part::Dup => {
                let val = self.peek_value()?;
                self.push_value(val);
            }
            Part::Swap => {
                let a = self.pop_value()?;
                let b = self.pop_value()?;
                self.push_value(a);
                self.push_value(b);
            }
            Part::RotateThree => {
                let a = self.pop_value()?;
                let b = self.pop_value()?;
                let c = self.pop_value()?;
                self.push_value(b);
                self.push_value(a);
                self.push_value(c);
            }
            Part::Pick(depth) => {
                let len = self.stack.len();
                if *depth < len {
                    let val = self.stack[len - 1 - *depth];
                    self.push_value(val);
                } else {
                    return Err(format!("Stack underflow on Pick({})", depth));
                }
            }
            Part::Roll(depth) => {
                let len = self.stack.len();
                if *depth < len {
                    let val = self.stack.remove(len - 1 - *depth);
                    self.push_value(val);
                } else {
                    return Err(format!("Stack underflow on Roll({})", depth));
                }
            }
            Part::Drop(count) => {
                for _ in 0..*count {
                    self.pop_value()?;
                }
            }
            Part::DupMultiple(count) => {
                let len = self.stack.len();
                if *count <= len {
                    let items: Vec<_> = self.stack[len - *count..].to_vec();
                    self.stack.extend(items);
                } else {
                    return Err(format!("Stack underflow on DupMultiple({})", count));
                }
            }
            Part::PushNull => {
                let ty = IntegerType::new(self.context, 64).into();
                let attr = IntegerAttribute::new(ty, 0);
                let op = arith::constant(self.context, attr.into(), location);
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::Nop => {}
            Part::Halt => {
                let ret_op = func::r#return(&[], location);
                block.append_operation(ret_op);
            }


            Part::Bool(val) => {
                let ty = IntegerType::new(self.context, 1).into();
                let attr = IntegerAttribute::new(ty, if *val { 1 } else { 0 });
                let op = arith::constant(self.context, attr.into(), location);
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::String(_) => {}
            Part::Print => {
                let callee = FlatSymbolRefAttribute::new(self.context, "iris_print");
                let args = if let Ok(val) = self.peek_value() {
                    vec![val]
                } else {
                    vec![]
                };
                let call_op = func::call(self.context, callee, &args, &[], location);
                block.append_operation(call_op);
            }


            Part::Integer(size, val, _unsigned) => {
                let ty = self.size_to_integer_type(*size);
                let attr = IntegerAttribute::new(ty, *val as i64);
                let op = arith::constant(self.context, attr.into(), location);
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::CalculateInteger(_size, op) => {
                let rhs = self.pop_value()?;
                let lhs = self.pop_value()?;
                let operation = match op {
                    IntegerCalculateType::Add => arith::addi(lhs, rhs, location),
                    IntegerCalculateType::Sub => arith::subi(lhs, rhs, location),
                    IntegerCalculateType::Mul => arith::muli(lhs, rhs, location),
                    IntegerCalculateType::Div(unsigned) => {
                        if *unsigned {
                            arith::divui(lhs, rhs, location)
                        } else {
                            arith::divsi(lhs, rhs, location)
                        }
                    }
                    IntegerCalculateType::Rem(unsigned) => {
                        if *unsigned {
                            arith::remui(lhs, rhs, location)
                        } else {
                            arith::remsi(lhs, rhs, location)
                        }
                    }
                };
                let op_ref = block.append_operation(operation);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::ComparisonInteger(cmp_type, unsigned, swap) => {
                let (lhs, rhs) = if *swap {
                    let r = self.pop_value()?;
                    let l = self.pop_value()?;
                    (r, l)
                } else {
                    let r = self.pop_value()?;
                    let l = self.pop_value()?;
                    (l, r)
                };
                let predicate = match cmp_type {
                    IntegerComparisonType::Equal => CmpiPredicate::Eq,
                    IntegerComparisonType::NotEqual => CmpiPredicate::Ne,
                    IntegerComparisonType::Greater => {
                        if *unsigned { CmpiPredicate::Ugt } else { CmpiPredicate::Sgt }
                    }
                    IntegerComparisonType::GreaterEqual => {
                        if *unsigned { CmpiPredicate::Uge } else { CmpiPredicate::Sge }
                    }
                    IntegerComparisonType::Less => {
                        if *unsigned { CmpiPredicate::Ult } else { CmpiPredicate::Slt }
                    }
                    IntegerComparisonType::LessEqual => {
                        if *unsigned { CmpiPredicate::Ule } else { CmpiPredicate::Sle }
                    }
                };
                let op = arith::cmpi(self.context, predicate, lhs, rhs, location);
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::ResizeInteger(from_size, to_size, signed) => {
                let val = self.pop_value()?;
                let from_bits = *from_size as u32;
                let to_bits = *to_size as u32;
                let to_ty = self.size_to_integer_type(*to_size);

                if to_bits > from_bits {
                    let op = if *signed {
                        arith::extsi(val, to_ty, location)
                    } else {
                        arith::extui(val, to_ty, location)
                    };
                    let op_ref = block.append_operation(op);
                    self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
                } else if to_bits < from_bits {
                    let op = arith::trunci(val, to_ty, location);
                    let op_ref = block.append_operation(op);
                    self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
                } else {
                    self.push_value(val);
                }
            }


            Part::Decimal(size, val) => {
                let ty = self.size_to_float_type(*size)?;
                let attr = FloatAttribute::new(self.context, ty, *val);
                let op = arith::constant(self.context, attr.into(), location);
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::CalculateDecimal(_size, op) => {
                let rhs = self.pop_value()?;
                let lhs = self.pop_value()?;
                let operation = match op {
                    DecimalCalculateType::Add => arith::addf(lhs, rhs, location),
                    DecimalCalculateType::Sub => arith::subf(lhs, rhs, location),
                    DecimalCalculateType::Mul => arith::mulf(lhs, rhs, location),
                    DecimalCalculateType::Div => arith::divf(lhs, rhs, location),
                    DecimalCalculateType::Rem => arith::remf(lhs, rhs, location),
                };
                let op_ref = block.append_operation(operation);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::DecimalNegative => {
                let val = self.pop_value()?;
                let op = arith::negf(val, location);
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::ComparisonDecimal(cmp_type, swap) => {
                let (lhs, rhs) = if *swap {
                    let r = self.pop_value()?;
                    let l = self.pop_value()?;
                    (r, l)
                } else {
                    let r = self.pop_value()?;
                    let l = self.pop_value()?;
                    (l, r)
                };
                let predicate = match cmp_type {
                    DecimalComparisonType::Equal => CmpfPredicate::Oeq,
                    DecimalComparisonType::NotEqual => CmpfPredicate::One,
                    DecimalComparisonType::Greater => CmpfPredicate::Ogt,
                    DecimalComparisonType::GreaterEqual => CmpfPredicate::Oge,
                    DecimalComparisonType::Less => CmpfPredicate::Olt,
                    DecimalComparisonType::LessEqual => CmpfPredicate::Ole,
                };
                let op = arith::cmpf(self.context, predicate, lhs, rhs, location);
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::ResizeDecimal(to_f64) => {
                let val = self.pop_value()?;
                let to_ty = if *to_f64 {
                    Type::float64(self.context)
                } else {
                    Type::float32(self.context)
                };
                let op = if *to_f64 {
                    arith::extf(val, to_ty, location)
                } else {
                    arith::truncf(val, to_ty, location)
                };
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }


            Part::UnaryMath(size, math_op) => {
                let val = self.pop_value()?;
                match math_op {
                    UnaryMathType::Abs => {
                        let zero_attr = IntegerAttribute::new(val.r#type(), 0);
                        let zero_op = arith::constant(self.context, zero_attr.into(), location);
                        let zero_val = block.append_operation(zero_op).result(0).unwrap().into();

                        let neg_op = arith::subi(zero_val, val, location);
                        let neg_val = block.append_operation(neg_op).result(0).unwrap().into();

                        let cmp_op = arith::cmpi(self.context, CmpiPredicate::Slt, val, zero_val, location);
                        let is_neg = block.append_operation(cmp_op).result(0).unwrap().into();

                        let sel_op = arith::select(is_neg, neg_val, val, location);
                        let res = block.append_operation(sel_op).result(0).unwrap().into();
                        self.push_value(res);
                    }
                    UnaryMathType::Negate => {
                        let zero_attr = IntegerAttribute::new(val.r#type(), 0);
                        let zero_op = arith::constant(self.context, zero_attr.into(), location);
                        let zero_val = block.append_operation(zero_op).result(0).unwrap().into();
                        let op = arith::subi(zero_val, val, location);
                        let res = block.append_operation(op).result(0).unwrap().into();
                        self.push_value(res);
                    }
                    UnaryMathType::Increment => {
                        let one_attr = IntegerAttribute::new(val.r#type(), 1);
                        let one_op = arith::constant(self.context, one_attr.into(), location);
                        let one_val = block.append_operation(one_op).result(0).unwrap().into();
                        let op = arith::addi(val, one_val, location);
                        let res = block.append_operation(op).result(0).unwrap().into();
                        self.push_value(res);
                    }
                    UnaryMathType::Decrement => {
                        let one_attr = IntegerAttribute::new(val.r#type(), 1);
                        let one_op = arith::constant(self.context, one_attr.into(), location);
                        let one_val = block.append_operation(one_op).result(0).unwrap().into();
                        let op = arith::subi(val, one_val, location);
                        let res = block.append_operation(op).result(0).unwrap().into();
                        self.push_value(res);
                    }
                    UnaryMathType::Sqrt => {
                        self.push_value(val);
                        self.emit_runtime_call(block, "iris_math_sqrt", 1, true)?;
                    }
                    UnaryMathType::Floor => {
                        self.push_value(val);
                        self.emit_runtime_call(block, "iris_math_floor", 1, true)?;
                    }
                    UnaryMathType::Ceil => {
                        self.push_value(val);
                        self.emit_runtime_call(block, "iris_math_ceil", 1, true)?;
                    }
                    UnaryMathType::Round => {
                        self.push_value(val);
                        self.emit_runtime_call(block, "iris_math_round", 1, true)?;
                    }
                    UnaryMathType::Truncate => {
                        self.push_value(val);
                        self.emit_runtime_call(block, "iris_math_trunc", 1, true)?;
                    }
                }
            }
            Part::FusedMultiplyAdd(_size) => {
                let c = self.pop_value()?;
                let b = self.pop_value()?;
                let a = self.pop_value()?;
                let mul_op = arith::mulf(a, b, location);
                let mul_res = block.append_operation(mul_op).result(0).unwrap().into();
                let add_op = arith::addf(mul_res, c, location);
                let res = block.append_operation(add_op).result(0).unwrap().into();
                self.push_value(res);
            }


            Part::IntegerToDecimal(unsigned, to_size) => {
                let val = self.pop_value()?;
                let to_ty = self.size_to_float_type(*to_size)?;
                let op = if *unsigned {
                    arith::uitofp(val, to_ty, location)
                } else {
                    arith::sitofp(val, to_ty, location)
                };
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::DecimalToInteger(unsigned, to_size) => {
                let val = self.pop_value()?;
                let to_ty = self.size_to_integer_type(*to_size);
                let op = if *unsigned {
                    arith::fptoui(val, to_ty, location)
                } else {
                    arith::fptosi(val, to_ty, location)
                };
                let op_ref = block.append_operation(op);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }


            Part::BinaryOperation(op) => {
                let rhs = self.pop_value()?;
                let lhs = self.pop_value()?;
                let operation = match op {
                    BinaryOperationType::LShift => arith::shli(lhs, rhs, location),
                    BinaryOperationType::RShift(unsigned) => {
                        if *unsigned {
                            arith::shrui(lhs, rhs, location)
                        } else {
                            arith::shrsi(lhs, rhs, location)
                        }
                    }
                    BinaryOperationType::And => arith::andi(lhs, rhs, location),
                    BinaryOperationType::Or => arith::ori(lhs, rhs, location),
                    BinaryOperationType::Xor => arith::xori(lhs, rhs, location),
                };
                let op_ref = block.append_operation(operation);
                self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
            }
            Part::BitwiseNot(_size) => {
                let val = self.pop_value()?;
                let ty = IntegerType::new(self.context, 64).into();
                let minus_one = block.append_operation(arith::constant(self.context, IntegerAttribute::new(ty, -1).into(), location)).result(0).unwrap().into();
                let op = arith::xori(val, minus_one, location);
                let res = block.append_operation(op).result(0).unwrap().into();
                self.push_value(res);
            }
            Part::Bitcast(_from, _to) => {

            }
            Part::MemCopy { size } => {
                let size_ty = IntegerType::new(self.context, 64).into();
                let size_val = block.append_operation(arith::constant(self.context, IntegerAttribute::new(size_ty, *size as i64).into(), location)).result(0).unwrap().into();
                self.push_value(size_val);
                self.emit_runtime_call(block, "iris_memcpy", 3, false)?;
            }
            Part::MemSet { value, size } => {
                let val_ty = IntegerType::new(self.context, 64).into();
                let val = block.append_operation(arith::constant(self.context, IntegerAttribute::new(val_ty, *value as i64).into(), location)).result(0).unwrap().into();
                let size_val = block.append_operation(arith::constant(self.context, IntegerAttribute::new(val_ty, *size as i64).into(), location)).result(0).unwrap().into();
                self.push_value(val);
                self.push_value(size_val);
                self.emit_runtime_call(block, "iris_memset", 3, false)?;
            }


            Part::ReturnInteger => {
                let return_op = if let Some(val) = self.stack.pop() {
                    func::r#return(&[val], location)
                } else {
                    func::r#return(&[], location)
                };
                block.append_operation(return_op);
            }
            Part::ReturnVoid => {
                let return_op = func::r#return(&[], location);
                block.append_operation(return_op);
            }
            Part::PushLocal(name) => {
                let val = self
                    .locals
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("Local variable '{}' not found", name))?;
                self.push_value(val);
            }
            Part::StoreLocal(name) => {
                let val = self.pop_value()?;
                self.locals.insert(name.clone(), val);
            }
            Part::PushGlobal(name) => {
                let val = self
                    .globals
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("Global variable '{}' not found", name))?;
                self.push_value(val);
            }
            Part::StoreGlobal(name) => {
                let val = self.pop_value()?;
                self.globals.insert(name.clone(), val);
            }
            Part::Call(name, args) => {
                let prev_stack_len = self.stack.len();
                for arg_part in args {
                    self.lower_part(block, arg_part)?;
                }
                let call_args: Vec<Value<'c, 'a>> = self.stack.drain(prev_stack_len..).collect();
                let callee = FlatSymbolRefAttribute::new(self.context, name);
                let call_op = func::call(self.context, callee, &call_args, &[], location);
                let op_ref = block.append_operation(call_op);
                if op_ref.result_count() > 0 {
                    self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
                }
            }
            Part::TailCall(name, args) => {
                let prev_stack_len = self.stack.len();
                for arg_part in args {
                    self.lower_part(block, arg_part)?;
                }
                let call_args: Vec<Value<'c, 'a>> = self.stack.drain(prev_stack_len..).collect();
                let callee = FlatSymbolRefAttribute::new(self.context, name);
                let call_op = func::call(self.context, callee, &call_args, &[], location);
                let op_ref = block.append_operation(call_op);
                let ret_op = if op_ref.result_count() > 0 {
                    let val: Value<'c, 'a> = op_ref.result(0).map_err(|e| e.to_string())?.into();
                    func::r#return(&[val], location)
                } else {
                    func::r#return(&[], location)
                };
                block.append_operation(ret_op);
            }
            Part::InvokeNative(name, args) => {
                let prev_stack_len = self.stack.len();
                for arg_part in args {
                    self.lower_part(block, arg_part)?;
                }
                let call_args: Vec<Value<'c, 'a>> = self.stack.drain(prev_stack_len..).collect();
                let callee = FlatSymbolRefAttribute::new(self.context, name);
                let call_op = func::call(self.context, callee, &call_args, &[], location);
                let op_ref = block.append_operation(call_op);
                if op_ref.result_count() > 0 {
                    self.push_value(op_ref.result(0).map_err(|e| e.to_string())?.into());
                }
            }


            Part::CreateArray { size, .. } => {
                let size_ty = IntegerType::new(self.context, 64).into();
                let size_attr = IntegerAttribute::new(size_ty, *size as i64);
                let size_op = arith::constant(self.context, size_attr.into(), location);
                let size_val = block.append_operation(size_op).result(0).unwrap().into();
                self.push_value(size_val);
                self.emit_runtime_call(block, "iris_create_array", 1, true)?;
            }
            Part::GetArrayElement => {
                self.emit_runtime_call(block, "iris_array_get", 2, true)?;
            }
            Part::SetArrayElement => {
                self.emit_runtime_call(block, "iris_array_set", 3, false)?;
            }
            Part::GetArrayLength => {
                self.emit_runtime_call(block, "iris_array_length", 1, true)?;
            }
            Part::ResizeArray => {
                self.emit_runtime_call(block, "iris_array_resize", 2, false)?;
            }
            Part::CreateMap { capacity } => {
                let cap_ty = IntegerType::new(self.context, 64).into();
                let cap_attr = IntegerAttribute::new(cap_ty, *capacity as i64);
                let cap_op = arith::constant(self.context, cap_attr.into(), location);
                let cap_val = block.append_operation(cap_op).result(0).unwrap().into();
                self.push_value(cap_val);
                self.emit_runtime_call(block, "iris_create_map", 1, true)?;
            }
            Part::MapGet => {
                self.emit_runtime_call(block, "iris_map_get", 2, true)?;
            }
            Part::MapSet => {
                self.emit_runtime_call(block, "iris_map_set", 3, false)?;
            }
            Part::MapContainsKey => {
                self.emit_runtime_call(block, "iris_map_contains_key", 2, true)?;
            }
            Part::MapRemoveKey => {
                self.emit_runtime_call(block, "iris_map_remove_key", 2, true)?;
            }
            Part::MapGetOrDefault => {
                self.emit_runtime_call(block, "iris_map_get_or_default", 3, true)?;
            }
            Part::AllocateSlice => {
                self.emit_runtime_call(block, "iris_allocate_slice", 3, true)?;
            }


            Part::DefineStruct { name, size, align, .. } => {
                let size_ty = IntegerType::new(self.context, 64).into();
                let size_attr = IntegerAttribute::new(size_ty, *size as i64);
                let align_attr = IntegerAttribute::new(size_ty, *align as i64);
                let size_val = block.append_operation(arith::constant(self.context, size_attr.into(), location)).result(0).unwrap().into();
                let align_val = block.append_operation(arith::constant(self.context, align_attr.into(), location)).result(0).unwrap().into();
                self.push_value(size_val);
                self.push_value(align_val);
                self.emit_runtime_call(block, &format!("iris_register_struct_{}", name), 2, false)?;
            }
            Part::AllocStackStruct(struct_name) => {
                self.emit_runtime_call(block, &format!("iris_alloca_struct_{}", struct_name), 0, true)?;
            }
            Part::AllocStruct(struct_name) => {
                self.emit_runtime_call(block, &format!("iris_alloc_struct_{}", struct_name), 0, true)?;
            }
            Part::GetStructField { offset, .. } => {
                let offset_ty = IntegerType::new(self.context, 64).into();
                let offset_attr = IntegerAttribute::new(offset_ty, *offset as i64);
                let offset_val = block.append_operation(arith::constant(self.context, offset_attr.into(), location)).result(0).unwrap().into();
                self.push_value(offset_val);
                self.emit_runtime_call(block, "iris_get_struct_field_raw", 2, true)?;
            }
            Part::SetStructField { offset, .. } => {
                let offset_ty = IntegerType::new(self.context, 64).into();
                let offset_attr = IntegerAttribute::new(offset_ty, *offset as i64);
                let offset_val = block.append_operation(arith::constant(self.context, offset_attr.into(), location)).result(0).unwrap().into();
                self.push_value(offset_val);
                self.emit_runtime_call(block, "iris_set_struct_field_raw", 3, false)?;
            }
            Part::GetStructFieldPtr { offset, .. } => {
                let ptr = self.pop_value()?;
                let offset_ty = IntegerType::new(self.context, 64).into();
                let offset_attr = IntegerAttribute::new(offset_ty, *offset as i64);
                let offset_val = block.append_operation(arith::constant(self.context, offset_attr.into(), location)).result(0).unwrap().into();
                let add_op = arith::addi(ptr, offset_val, location);
                let res = block.append_operation(add_op).result(0).unwrap().into();
                self.push_value(res);
            }
            Part::CallMethod { struct_name, method_name, args } => {
                let prev_stack_len = self.stack.len();
                for arg_part in args {
                    self.lower_part(block, arg_part)?;
                }
                let arg_count = self.stack.len() - prev_stack_len + 1;

                let mangled_name = format!("{}__{}", struct_name, method_name);
                self.emit_runtime_call(block, &mangled_name, arg_count, true)?;
            }
            Part::AllocRaw { size, align } => {
                let size_ty = IntegerType::new(self.context, 64).into();
                let size_attr = IntegerAttribute::new(size_ty, *size as i64);
                let align_attr = IntegerAttribute::new(size_ty, *align as i64);
                let size_val = block.append_operation(arith::constant(self.context, size_attr.into(), location)).result(0).unwrap().into();
                let align_val = block.append_operation(arith::constant(self.context, align_attr.into(), location)).result(0).unwrap().into();
                self.push_value(size_val);
                self.push_value(align_val);
                self.emit_runtime_call(block, "iris_alloc_raw", 2, true)?;
            }
            Part::FreeRaw => {
                self.emit_runtime_call(block, "iris_free_raw", 1, false)?;
            }
            Part::LoadPtr(_size) => {
                self.emit_runtime_call(block, "iris_load_ptr", 1, true)?;
            }
            Part::StorePtr(_size) => {
                self.emit_runtime_call(block, "iris_store_ptr", 2, false)?;
            }
            Part::OffsetPtr(offset) => {
                let ptr = self.pop_value()?;
                let offset_ty = IntegerType::new(self.context, 64).into();
                let offset_attr = IntegerAttribute::new(offset_ty, *offset);
                let offset_val = block.append_operation(arith::constant(self.context, offset_attr.into(), location)).result(0).unwrap().into();
                let add_op = arith::addi(ptr, offset_val, location);
                let res = block.append_operation(add_op).result(0).unwrap().into();
                self.push_value(res);
            }


            Part::AtomicAdd(_) => {
                self.emit_runtime_call(block, "iris_atomic_add", 2, true)?;
            }
            Part::AtomicSub(_) => {
                self.emit_runtime_call(block, "iris_atomic_sub", 2, true)?;
            }
            Part::AtomicCas(_) => {
                self.emit_runtime_call(block, "iris_atomic_cas", 3, true)?;
            }
            Part::EnterMonitor => {
                self.emit_runtime_call(block, "iris_enter_monitor", 1, false)?;
            }
            Part::ExitMonitor => {
                self.emit_runtime_call(block, "iris_exit_monitor", 1, false)?;
            }
            Part::YieldThread => {
                self.emit_runtime_call(block, "iris_yield_thread", 0, false)?;
            }


            Part::CallWithCache { method, args } | Part::MegamorphicCall { method, args } => {
                let prev_stack_len = self.stack.len();
                for arg_part in args {
                    self.lower_part(block, arg_part)?;
                }
                let arg_count = self.stack.len() - prev_stack_len;
                self.emit_runtime_call(block, &format!("iris_ic_call_{}", method), arg_count, true)?;
            }
            Part::GetPropertyWithCache(prop) => {
                self.emit_runtime_call(block, &format!("iris_ic_get_{}", prop), 1, true)?;
            }
            Part::SetPropertyWithCache(prop) => {
                self.emit_runtime_call(block, &format!("iris_ic_set_{}", prop), 2, false)?;
            }


            Part::Throw => {
                self.emit_runtime_call(block, "iris_throw_exception", 1, false)?;
            }
            Part::Try(catch_label) => {
                self.emit_runtime_call(block, &format!("iris_push_try_frame_{}", catch_label), 0, false)?;
            }
            Part::EndTry => {
                self.emit_runtime_call(block, "iris_pop_try_frame", 0, false)?;
            }


            Part::KernelLaunch { name, .. } => {
                self.emit_runtime_call(block, &format!("iris_gpu_launch_{}", name), 0, false)?;
            }
            Part::DeviceMemoryAlloc(name, size) => {
                let size_ty = IntegerType::new(self.context, 64).into();
                let size_attr = IntegerAttribute::new(size_ty, *size as i64);
                let size_op = arith::constant(self.context, size_attr.into(), location);
                let size_val = block.append_operation(size_op).result(0).unwrap().into();
                self.push_value(size_val);
                self.emit_runtime_call(block, &format!("iris_gpu_alloc_{}", name), 1, true)?;
            }
            Part::CopyToDevice(name) => {
                self.emit_runtime_call(block, &format!("iris_gpu_copy_to_{}", name), 1, false)?;
            }
            Part::CopyFromDevice(name) => {
                self.emit_runtime_call(block, &format!("iris_gpu_copy_from_{}", name), 1, false)?;
            }


            Part::Label(_)
            | Part::Jump(_)
            | Part::JumpIfFalse(_)
            | Part::JumpIfTrue(_)
            | Part::JumpIfNull(_)
            | Part::JumpIfNonNull(_)
            | Part::TableSwitch { .. } => {}
        }
        Ok(())
    }
}
