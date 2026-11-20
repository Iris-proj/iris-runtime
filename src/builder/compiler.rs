use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::types::{FloatType, FunctionType, IntType, PointerType, VoidType};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use inkwell::values::{FloatValue, FunctionValue, IntValue};
use crate::builder::paper::Paper;
use crate::builder::part::{BinaryOperationType, DecimalCalculateType, DecimalComparisonType, IntegerCalculateType, IntegerComparisonType, Part, Size, UnaryMathType};
use crate::vm::chunk::{Chunk, ChunkWriter};
use crate::vm::engine::IrisEngine;
use crate::vm::opcode::OpCode;
use crate::vm::value::Value;

pub type Func = unsafe extern "C" fn(iris_engine: *mut IrisEngine);

#[allow(unused)]
pub struct RuntimeFunctionCompiler<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,

    vm_type: PointerType<'ctx>,
    function_type: FunctionType<'ctx>,
    null_type: VoidType<'ctx>,
    bool_type: IntType<'ctx>,
    i8_type: IntType<'ctx>,
    i16_type: IntType<'ctx>,
    i32_type: IntType<'ctx>,
    i64_type: IntType<'ctx>,
    i128_type: IntType<'ctx>,
    f32_type: FloatType<'ctx>,
    f64_type: FloatType<'ctx>,

    print_fn: FunctionValue<'ctx>,
    push_i32_fn: FunctionValue<'ctx>,
    push_i64_fn: FunctionValue<'ctx>,
    get_global_id_fn: FunctionValue<'ctx>,
    get_thread_idx_fn: FunctionValue<'ctx>,
    get_block_idx_fn: FunctionValue<'ctx>,
    spawn_detached_fn: FunctionValue<'ctx>,
    push_str_fn: FunctionValue<'ctx>,
}

impl<'ctx> RuntimeFunctionCompiler<'ctx> {
    pub fn new(context: &'ctx Context, module: Module<'ctx>) -> Self {
        let i8_type = context.i8_type();
        let i64_type = context.i64_type();
        let null_type = context.void_type();
        let vm_type = context.ptr_type(AddressSpace::from(0));
        let function_type =  i8_type.fn_type(&[vm_type.into()], false);
        let print_fn = module.get_function("iris_print_logic")
            .unwrap_or_else(|| module.add_function("iris_print_logic", function_type, None));
        let push_i32_type = null_type.fn_type(&[vm_type.into(), context.i32_type().into()], false);
        let push_i32_fn = module.get_function("iris_push_i32")
            .unwrap_or_else(|| module.add_function("iris_push_i32", push_i32_type, None));
        let push_i64_type = null_type.fn_type(&[vm_type.into(), i64_type.into()], false);
        let push_i64_fn = module.get_function("iris_push_i64")
            .unwrap_or_else(|| module.add_function("iris_push_i64", push_i64_type, None));

        let get_id_type = i64_type.fn_type(&[vm_type.into()], false);
        let get_global_id_fn = module.get_function("iris_get_global_id")
            .unwrap_or_else(|| module.add_function("iris_get_global_id", get_id_type, None));
        let get_thread_idx_fn = module.get_function("iris_get_thread_idx")
            .unwrap_or_else(|| module.add_function("iris_get_thread_idx", get_id_type, None));
        let get_block_idx_fn = module.get_function("iris_get_block_idx")
            .unwrap_or_else(|| module.add_function("iris_get_block_idx", get_id_type, None));

        let spawn_type = null_type.fn_type(&[
            context.ptr_type(AddressSpace::from(0)).into(),
            i64_type.into(),
        ], false);
        let spawn_detached_fn = module.get_function("iris_spawn_detached")
            .unwrap_or_else(|| module.add_function("iris_spawn_detached", spawn_type, None));

        let push_str_type = null_type.fn_type(&[
            vm_type.into(),
            context.ptr_type(AddressSpace::from(0)).into(),
            i64_type.into(),
        ], false);
        let push_str_fn = module.get_function("iris_push_str")
            .unwrap_or_else(|| module.add_function("iris_push_str", push_str_type, None));

        RuntimeFunctionCompiler {
            context,
            module,
            vm_type,
            function_type,
            null_type,
            bool_type: context.bool_type(),
            i8_type,
            i16_type: context.i16_type(),
            i32_type: context.i32_type(),
            i64_type,
            i128_type: context.i128_type(),
            f32_type: context.f32_type(),
            f64_type: context.f64_type(),

            print_fn,
            push_i32_fn,
            push_i64_fn,
            get_global_id_fn,
            get_thread_idx_fn,
            get_block_idx_fn,
            spawn_detached_fn,
            push_str_fn,
        }
    }

    pub fn setup_ee(&self, ee: &ExecutionEngine){
        ee.add_global_mapping(&self.print_fn, iris_print_logic as *const () as usize);
        ee.add_global_mapping(&self.push_i32_fn, iris_push_i32_logic as *const () as usize);
        ee.add_global_mapping(&self.push_i64_fn, iris_push_i64_logic as *const () as usize);
        ee.add_global_mapping(&self.get_global_id_fn, iris_get_global_id as *const () as usize);
        ee.add_global_mapping(&self.get_thread_idx_fn, iris_get_thread_idx as *const () as usize);
        ee.add_global_mapping(&self.get_block_idx_fn, iris_get_block_idx as *const () as usize);
        ee.add_global_mapping(&self.spawn_detached_fn, iris_spawn_detached as *const () as usize);
        ee.add_global_mapping(&self.push_str_fn, iris_push_str as *const () as usize);
    }

    pub fn compile_paper_function(
        &self,
        paper: &Paper,
    ) -> Option<FunctionValue<'ctx>> {
        let mut int_values: Vec<IntValue> = vec![];
        let mut float_values: Vec<FloatValue> = vec![];
        let mut locals_map: std::collections::HashMap<String, inkwell::values::PointerValue<'ctx>> = std::collections::HashMap::new();

        let function = if let Some(existing) = self.module.get_function(paper.name.as_str()) {
            if existing.count_basic_blocks() > 0 {
                return Some(existing);
            }
            existing
        } else {
            self.module.add_function(paper.name.as_str(), self.function_type, None)
        };
        let basic_block = self.context.append_basic_block(function, "entry");


        let entry_builder = self.context.create_builder();
        entry_builder.position_at_end(basic_block);
        for part in &paper.parts {
            match part {
                Part::PushLocal(name) | Part::StoreLocal(name) => {
                    locals_map.entry(name.clone()).or_insert_with(|| {
                        entry_builder.build_alloca(self.i64_type, name).unwrap()
                    });
                }
                _ => {}
            }
        }

        let mut label_blocks: std::collections::HashMap<String, inkwell::basic_block::BasicBlock<'ctx>> = std::collections::HashMap::new();
        for part in &paper.parts {
            if let Part::Label(name) = part {
                if !label_blocks.contains_key(name) {
                    let bb = self.context.append_basic_block(function, name.as_str());
                    label_blocks.insert(name.clone(), bb);
                }
            }
        }

        let builder = self.context.create_builder();
        builder.position_at_end(basic_block);

        let engine_ptr = function.get_first_param()?.into_pointer_value();

        for part in paper.parts.clone() {
            match part {
                Part::Print => {
                    if let Some(val) = int_values.pop() {
                        if val.get_type() == self.i64_type {
                            let _ = builder.build_call(self.push_i64_fn, &[engine_ptr.into(), val.into()], "push_i64_call");
                        } else {
                            let val_i32 = if val.get_type() == self.i32_type {
                                val
                            } else {
                                builder.build_int_truncate(val, self.i32_type, "trunc_i32").unwrap()
                            };
                            let _ = builder.build_call(self.push_i32_fn, &[engine_ptr.into(), val_i32.into()], "push_i32_call");
                        }
                    }
                    let _ = builder.build_call(self.print_fn, &[engine_ptr.into()], "print_call");
                },
                Part::String(s) => {
                    let str_val = builder.build_global_string_ptr(&s, "str_lit").unwrap();
                    let len_val = self.i64_type.const_int(s.len() as u64, false);
                    let _ = builder.build_call(self.push_str_fn, &[engine_ptr.into(), str_val.as_pointer_value().into(), len_val.into()], "push_str_call");
                },
                Part::Integer(size, value, unsigned) => {
                    match size {
                        Size::S8 => {
                            int_values.push(self.i8_type.const_int(value, unsigned));
                        },
                        Size::S16 => {
                            int_values.push(self.i16_type.const_int(value, unsigned));
                        },
                        Size::S32 => {
                            int_values.push(self.i32_type.const_int(value, unsigned));
                        },
                        Size::S64 => {
                            int_values.push(self.i64_type.const_int(value, unsigned));
                        },
                        Size::S128 => {
                            int_values.push(self.i128_type.const_int(value, unsigned));
                        }
                    }
                },
                Part::CalculateInteger(_size, mode) => {
                    let rhs = int_values.pop().unwrap();
                    let lhs = int_values.pop().unwrap();
                    match mode {
                        IntegerCalculateType::Add => {
                            int_values.push(builder.build_int_add(lhs, rhs, "add_tmp").unwrap())
                        },
                        IntegerCalculateType::Sub => {
                            int_values.push(builder.build_int_sub(lhs, rhs, "sub_tmp").unwrap())
                        },
                        IntegerCalculateType::Mul => {
                            int_values.push(builder.build_int_mul(lhs, rhs, "mul_tmp").unwrap())
                        },
                        IntegerCalculateType::Div(unsigned) => {
                            if unsigned {
                                int_values.push(builder.build_int_unsigned_div(lhs, rhs, "div_tmp").unwrap())
                            }
                            else {
                                int_values.push(builder.build_int_signed_div(lhs, rhs, "div_tmp").unwrap())
                            }
                        },
                        IntegerCalculateType::Rem(unsigned) => {
                            if unsigned {
                                int_values.push(builder.build_int_unsigned_rem(lhs, rhs, "rem_tmp").unwrap())
                            }
                            else {
                                int_values.push(builder.build_int_signed_rem(lhs, rhs, "rem_tmp").unwrap())
                            }
                        }
                    }
                },
                Part::ComparisonInteger(mode, unsigned, _size) => {
                    let rhs = int_values.pop().unwrap();
                    let lhs = int_values.pop().unwrap();
                    match mode {
                        IntegerComparisonType::Equal => {
                            int_values.push(builder.build_int_compare(IntPredicate::EQ, lhs, rhs, "eq_tmp").unwrap());
                        },
                        IntegerComparisonType::NotEqual => {
                            int_values.push(builder.build_int_compare(IntPredicate::NE, lhs, rhs, "ne_tmp").unwrap());
                        },
                        IntegerComparisonType::Greater => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::UGT} else {IntPredicate::SGT}, lhs, rhs, "g_tmp").unwrap());
                        },
                        IntegerComparisonType::GreaterEqual => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::UGE} else {IntPredicate::SGE}, lhs, rhs, "ge_tmp").unwrap());
                        },
                        IntegerComparisonType::Less => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::ULT} else {IntPredicate::SLT}, lhs, rhs, "l_tmp").unwrap());
                        },
                        IntegerComparisonType::LessEqual => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::ULE} else {IntPredicate::SLE}, lhs, rhs, "le_tmp").unwrap());
                        }
                    }
                },
                Part::ResizeInteger(current_size, to_size, z_or_s) => {
                    let value = int_values.pop().unwrap();

                    fn size_to_type<'a>(size: Size, rfc: &RuntimeFunctionCompiler<'a>) -> IntType<'a> {
                        match size {
                            Size::S8 => { rfc.i8_type},
                            Size::S16 => { rfc.i16_type},
                            Size::S32 => { rfc.i32_type},
                            Size::S64 => { rfc.i64_type},
                            Size::S128 => { rfc.i128_type},
                        }
                    }
                    let to_type = size_to_type(to_size, self);


                    if current_size < to_size {
                        if z_or_s {
                            int_values.push(builder.build_int_z_extend(value, to_type, "tmp_rsz").unwrap())
                        }
                        else {
                            int_values.push(builder.build_int_s_extend(value, to_type, "tmp_rsz").unwrap())
                        }
                    }
                    else if current_size > to_size {
                        int_values.push(builder.build_int_truncate(value, to_type, "tmp_rsz").unwrap())
                    }
                    else {
                        int_values.push(value);
                    }
                },
                Part::IntegerToDecimal(unsigned, mode) => {
                    let value = int_values.pop().unwrap();

                    let float_type = match mode { Size::S32 => {self.f32_type}, Size::S64 => {self.f64_type}, _ => unreachable!()};

                    float_values.push(
                        if unsigned {builder.build_unsigned_int_to_float(value, float_type, "tmp_i2f").unwrap()}
                        else {builder.build_signed_int_to_float(value, float_type, "tmp_i2f").unwrap()})
                }
                Part::ReturnInteger => {
                    if let Some(val) = int_values.pop() {
                        if val.get_type() == self.i64_type {
                            let _ = builder.build_call(self.push_i64_fn, &[engine_ptr.into(), val.into()], "push_i64_call");
                        } else {
                            let val_i32 = if val.get_type() == self.i32_type {
                                val
                            } else {
                                builder.build_int_truncate(val, self.i32_type, "trunc_i32").unwrap()
                            };
                            let _ = builder.build_call(self.push_i32_fn, &[engine_ptr.into(), val_i32.into()], "push_i32_call");
                        }
                    }
                }
                Part::Decimal(size, value) => {
                    match size {
                        Size::S32 => {
                            float_values.push(self.f32_type.const_float(value));
                        },
                        Size::S64 => {
                            float_values.push(self.f64_type.const_float(value));
                        },
                        _ => {

                        }
                    }
                }
                Part::CalculateDecimal(_size, mode) => {
                    let rhs = float_values.pop().unwrap();
                    let lhs = float_values.pop().unwrap();

                    match mode {
                        DecimalCalculateType::Add => {
                            float_values.push(builder.build_float_add(lhs, rhs, "add_tmp").unwrap());
                        },
                        DecimalCalculateType::Sub => {
                            float_values.push(builder.build_float_sub(lhs, rhs, "sub_tmp").unwrap());
                        },
                        DecimalCalculateType::Mul => {
                            float_values.push(builder.build_float_mul(lhs, rhs, "mul_tmp").unwrap());
                        },
                        DecimalCalculateType::Div => {
                            float_values.push(builder.build_float_div(lhs, rhs, "div_tmp").unwrap());
                        },
                        DecimalCalculateType::Rem => {
                            float_values.push(builder.build_float_rem(lhs, rhs, "rem_tmp").unwrap());
                        }
                    }
                },
                Part::DecimalNegative => {
                    let value = float_values.pop().unwrap();
                    float_values.push(builder.build_float_neg(value, "neg_tmp").unwrap());
                },
                Part::ComparisonDecimal(mode, ordered) => {
                    let rhs = float_values.pop().unwrap();
                    let lhs = float_values.pop().unwrap();

                    match mode {
                        DecimalComparisonType::Equal => {
                            int_values.push(
                                builder.build_float_compare(
                                    if ordered { FloatPredicate::OEQ } else { FloatPredicate::UEQ },
                                    lhs,
                                    rhs,
                                    "eq_tmp",
                                ).unwrap()
                            );
                        }
                        DecimalComparisonType::NotEqual => {
                            int_values.push(
                                builder.build_float_compare(
                                    if ordered { FloatPredicate::ONE } else { FloatPredicate::UNE },
                                    lhs,
                                    rhs,
                                    "ne_tmp",
                                ).unwrap()
                            );
                        }
                        DecimalComparisonType::Greater => {
                            int_values.push(
                                builder.build_float_compare(
                                    if ordered { FloatPredicate::OGT } else { FloatPredicate::UGT },
                                    lhs,
                                    rhs,
                                    "gt_tmp",
                                ).unwrap()
                            );
                        }
                        DecimalComparisonType::GreaterEqual => {
                            int_values.push(
                                builder.build_float_compare(
                                    if ordered { FloatPredicate::OGE } else { FloatPredicate::UGE },
                                    lhs,
                                    rhs,
                                    "ge_tmp",
                                ).unwrap()
                            );
                        }
                        DecimalComparisonType::Less => {
                            int_values.push(
                                builder.build_float_compare(
                                    if ordered { FloatPredicate::OLT } else { FloatPredicate::ULT },
                                    lhs,
                                    rhs,
                                    "lt_tmp",
                                ).unwrap()
                            );
                        }
                        DecimalComparisonType::LessEqual => {
                            int_values.push(
                                builder.build_float_compare(
                                    if ordered { FloatPredicate::OLE } else { FloatPredicate::ULE },
                                    lhs,
                                    rhs,
                                    "le_tmp",
                                ).unwrap()
                            );
                        }
                    }
                },
                Part::ResizeDecimal(extend) => {
                    let value = float_values.pop().unwrap();

                    if extend {
                        float_values.push(builder.build_float_ext(value, self.f64_type, "tmp_ext").unwrap());
                    }
                    else {
                        float_values.push(builder.build_float_trunc(value, self.f32_type, "tmp_trc").unwrap());
                    }
                },
                Part::DecimalToInteger(unsigned, mode) => {
                    let value = float_values.pop().unwrap();

                    let int_type =
                        match mode {
                            Size::S8 => {self.i8_type},
                            Size::S16 => {self.i16_type},
                            Size::S32 => {self.i32_type},
                            Size::S64 => {self.i64_type},
                            Size::S128 => {self.i128_type},
                        };

                    int_values.push(
                        if unsigned {builder.build_float_to_unsigned_int(value, int_type, "tmp_i2f").unwrap()}
                        else {builder.build_float_to_signed_int(value, int_type, "tmp_i2f").unwrap()})
                }
                Part::BinaryOperation(mode) => {
                    let rhs = int_values.pop().unwrap();
                    let lhs = int_values.pop().unwrap();

                    match mode {
                        BinaryOperationType::LShift => {
                            int_values.push(builder.build_left_shift(lhs, rhs, "ls").unwrap());
                        },
                        BinaryOperationType::RShift(sg) => {
                            int_values.push(builder.build_right_shift(lhs, rhs, sg, "ls").unwrap());
                        },
                        BinaryOperationType::And => {
                            int_values.push(builder.build_and(lhs, rhs, "and_tmp").unwrap());
                        },
                        BinaryOperationType::Or => {
                            int_values.push(builder.build_or(lhs, rhs, "or_tmp").unwrap());
                        },
                        BinaryOperationType::Xor => {
                            int_values.push(builder.build_xor(lhs, rhs, "xor_tmp").unwrap());
                        }
                    }
                },
                Part::Bool(value) => {
                    int_values.push(self.bool_type.const_int(if value {1} else {0}, false));
                },
                Part::Pop => {
                    let _ = int_values.pop();
                }
                Part::Dup => {
                    if let Some(top) = int_values.last().cloned() {
                        int_values.push(top);
                    }
                }
                Part::Swap => {
                    let len = int_values.len();
                    if len >= 2 {
                        int_values.swap(len - 1, len - 2);
                    }
                }
                Part::RotateThree => {
                    let len = int_values.len();
                    if len >= 3 {
                        let c = int_values.pop().unwrap();
                        let b = int_values.pop().unwrap();
                        let a = int_values.pop().unwrap();
                        int_values.push(b);
                        int_values.push(c);
                        int_values.push(a);
                    }
                }
                Part::PushNull => {
                    int_values.push(self.i64_type.const_int(0, false));
                }
                Part::Nop => {}
                Part::Label(name) => {
                    if let Some(target_bb) = label_blocks.get(&name) {
                        if let Some(current_bb) = builder.get_insert_block() {
                            if current_bb.get_terminator().is_none() {
                                builder.build_unconditional_branch(*target_bb).ok();
                            }
                        }
                        builder.position_at_end(*target_bb);
                    }
                }
                Part::Jump(name) => {
                    if let Some(target_bb) = label_blocks.get(&name) {
                        builder.build_unconditional_branch(*target_bb).ok();
                        let dead_bb = self.context.append_basic_block(function, "after_jump");
                        builder.position_at_end(dead_bb);
                    }
                }
                Part::JumpIfTrue(name) => {
                    if let Some(val) = int_values.pop() {
                        if let Some(target_bb) = label_blocks.get(&name) {
                            let cond = if val.get_type() == self.bool_type {
                                val
                            } else {
                                let zero = val.get_type().const_int(0, false);
                                builder.build_int_compare(IntPredicate::NE, val, zero, "tobool").unwrap()
                            };
                            let fallthrough_bb = self.context.append_basic_block(function, "fallthrough");
                            builder.build_conditional_branch(cond, *target_bb, fallthrough_bb).ok();
                            builder.position_at_end(fallthrough_bb);
                        }
                    }
                }
                Part::JumpIfFalse(name) => {
                    if let Some(val) = int_values.pop() {
                        if let Some(target_bb) = label_blocks.get(&name) {
                            let is_true = if val.get_type() == self.bool_type {
                                val
                            } else {
                                let zero = val.get_type().const_int(0, false);
                                builder.build_int_compare(IntPredicate::NE, val, zero, "to_bool").unwrap()
                            };
                            let fallthrough_bb = self.context.append_basic_block(function, "fallthrough");
                            builder.build_conditional_branch(is_true, fallthrough_bb, *target_bb).ok();
                            builder.position_at_end(fallthrough_bb);
                        }
                    }
                }
                Part::Halt => break,
                Part::BitwiseNot(_) => {
                    if let Some(val) = int_values.pop() {
                        int_values.push(builder.build_not(val, "not_tmp").unwrap());
                    }
                }
                Part::ReturnVoid => {
                    if let Some(current_bb) = builder.get_insert_block() {
            if current_bb.get_terminator().is_none() {
                builder.build_return(Some(&self.i8_type.const_int(0, false))).ok();
            }
        }
                }
                Part::PushLocal(name) => {
                    let alloca = locals_map.entry(name.clone()).or_insert_with(|| {
                        let ptr = builder.build_alloca(self.i64_type, &name).unwrap();
                        if name == "global_id" {
                            if let Ok(call) = builder.build_call(self.get_global_id_fn, &[engine_ptr.into()], "call_gid") {
                                if let inkwell::values::ValueKind::Basic(val) = call.try_as_basic_value() {
                                    let _ = builder.build_store(ptr, val.into_int_value());
                                }
                            }
                        } else if name == "thread_idx" {
                            if let Ok(call) = builder.build_call(self.get_thread_idx_fn, &[engine_ptr.into()], "call_tidx") {
                                if let inkwell::values::ValueKind::Basic(val) = call.try_as_basic_value() {
                                    let _ = builder.build_store(ptr, val.into_int_value());
                                }
                            }
                        } else if name == "block_idx" {
                            if let Ok(call) = builder.build_call(self.get_block_idx_fn, &[engine_ptr.into()], "call_bidx") {
                                if let inkwell::values::ValueKind::Basic(val) = call.try_as_basic_value() {
                                    let _ = builder.build_store(ptr, val.into_int_value());
                                }
                            }
                        }
                        ptr
                    });
                    let loaded = builder.build_load(self.i64_type, *alloca, "loc_load").unwrap();
                    int_values.push(loaded.into_int_value());
                }
                Part::StoreLocal(name) => {
                    if let Some(val) = int_values.pop() {
                        let alloca = locals_map.entry(name).or_insert_with(|| {
                            builder.build_alloca(self.i64_type, "loc").unwrap()
                        });
                        builder.build_store(*alloca, val).unwrap();
                    }
                }
                Part::SpawnDetached(name) => {
                    let name_val = builder.build_global_string_ptr(&name, "str_spawn").unwrap();
                    let len_val = self.i64_type.const_int(name.len() as u64, false);
                    let _ = builder.build_call(self.spawn_detached_fn, &[name_val.as_pointer_value().into(), len_val.into()], "call_spawn");
                }
                Part::AllocStackStruct(_name) | Part::AllocStruct(_name) => {
                    let alloca = builder.build_alloca(self.i64_type, "struct_alloc").unwrap();
                    let ptr_int = builder.build_ptr_to_int(alloca, self.i64_type, "ptr_int").unwrap();
                    int_values.push(ptr_int);
                }
                Part::GetStructField { offset, size, .. } => {
                    let base = int_values.pop().unwrap_or_else(|| self.i64_type.const_int(0, false));
                    let base_ptr = builder.build_int_to_ptr(base, self.context.ptr_type(AddressSpace::from(0)), "base_ptr").unwrap();
                    let offset_val = self.i64_type.const_int(offset as u64, false);
                    let field_ptr = unsafe { builder.build_gep(self.i8_type, base_ptr, &[offset_val], "gep").unwrap() };
                    let ty = match size {
                        Size::S8 => self.i8_type,
                        Size::S16 => self.i16_type,
                        Size::S32 => self.i32_type,
                        _ => self.i64_type,
                    };
                    let loaded = builder.build_load(ty, field_ptr, "fld").unwrap();
                    int_values.push(loaded.into_int_value());
                }
                Part::SetStructField { offset, .. } => {
                    if let Some(val) = int_values.pop() {
                        if let Some(base) = int_values.last().cloned() {
                            let base_ptr = builder.build_int_to_ptr(base, self.context.ptr_type(AddressSpace::from(0)), "base_ptr").unwrap();
                            let offset_val = self.i64_type.const_int(offset as u64, false);
                            let field_ptr = unsafe { builder.build_gep(self.i8_type, base_ptr, &[offset_val], "gep").unwrap() };
                            builder.build_store(field_ptr, val).unwrap();
                        }
                    }
                }
                Part::GetStructFieldPtr { offset, .. } => {
                    if let Some(base) = int_values.pop() {
                        let offset_val = self.i64_type.const_int(offset as u64, false);
                        int_values.push(builder.build_int_add(base, offset_val, "ptr_add").unwrap());
                    }
                }
                Part::CallMethod { .. } => {}
                Part::UnaryMath(_size, math_type) => {
                    match math_type {
                        UnaryMathType::Abs => {
                            if let Some(val) = int_values.pop() {
                                let zero = val.get_type().const_int(0, false);
                                let neg = builder.build_int_neg(val, "neg").unwrap();
                                let is_neg = builder.build_int_compare(IntPredicate::SLT, val, zero, "is_neg").unwrap();
                                let abs = builder.build_select(is_neg, neg, val, "abs").unwrap();
                                int_values.push(abs.into_int_value());
                            }
                        }
                        UnaryMathType::Negate => {
                            if let Some(val) = int_values.pop() {
                                int_values.push(builder.build_int_neg(val, "neg").unwrap());
                            }
                        }
                        UnaryMathType::Increment => {
                            if let Some(val) = int_values.pop() {
                                let one = val.get_type().const_int(1, false);
                                int_values.push(builder.build_int_add(val, one, "inc").unwrap());
                            }
                        }
                        UnaryMathType::Decrement => {
                            if let Some(val) = int_values.pop() {
                                let one = val.get_type().const_int(1, false);
                                int_values.push(builder.build_int_sub(val, one, "dec").unwrap());
                            }
                        }
                        _ => {}
                    }
                }
                Part::FusedMultiplyAdd(_) => {
                    if float_values.len() >= 3 {
                        let c = float_values.pop().unwrap();
                        let b = float_values.pop().unwrap();
                        let a = float_values.pop().unwrap();
                        let mul = builder.build_float_mul(a, b, "fma_mul").unwrap();
                        float_values.push(builder.build_float_add(mul, c, "fma_add").unwrap());
                    }
                }
                _ => {}
            }
        }

        if let Some(current_bb) = builder.get_insert_block() {
            if current_bb.get_terminator().is_none() {
                builder.build_return(Some(&self.i8_type.const_int(0, false))).ok();
            }
        }

        Some(function)
    }


    pub fn compile_paper_function_pure(
        &self,
        paper: &Paper,
    ) -> Option<FunctionValue<'ctx>> {
        let mut int_values: Vec<IntValue> = vec![];
        let mut float_values: Vec<FloatValue> = vec![];
        let mut locals_map: std::collections::HashMap<String, inkwell::values::PointerValue<'ctx>> = std::collections::HashMap::new();


        let pure_fn_type = self.i64_type.fn_type(&[], false);
        let function = if let Some(existing) = self.module.get_function(paper.name.as_str()) {
            if existing.count_basic_blocks() > 0 {
                return Some(existing);
            }
            existing
        } else {
            self.module.add_function(paper.name.as_str(), pure_fn_type, None)
        };
        let basic_block = self.context.append_basic_block(function, "entry");


        let entry_builder = self.context.create_builder();
        entry_builder.position_at_end(basic_block);
        for part in &paper.parts {
            match part {
                Part::PushLocal(name) | Part::StoreLocal(name) => {
                    locals_map.entry(name.clone()).or_insert_with(|| {
                        entry_builder.build_alloca(self.i64_type, name).unwrap()
                    });
                }
                _ => {}
            }
        }

        let mut label_blocks: std::collections::HashMap<String, inkwell::basic_block::BasicBlock<'ctx>> = std::collections::HashMap::new();
        for part in &paper.parts {
            if let Part::Label(name) = part {
                if !label_blocks.contains_key(name) {
                    let bb = self.context.append_basic_block(function, name.as_str());
                    label_blocks.insert(name.clone(), bb);
                }
            }
        }

        let builder = self.context.create_builder();
        builder.position_at_end(basic_block);

        let ptr_type = self.context.ptr_type(AddressSpace::from(0));
        let puts_fn = self.module.get_function("puts")
            .unwrap_or_else(|| self.module.add_function("puts", self.i32_type.fn_type(&[ptr_type.into()], false), None));
        let printf_fn = self.module.get_function("printf")
            .unwrap_or_else(|| self.module.add_function("printf", self.i32_type.fn_type(&[ptr_type.into()], true), None));

        for part in paper.parts.clone() {
            match part {
                Part::Print => {
                    if let Some(val) = int_values.pop() {
                        let fmt = builder.build_global_string_ptr("%ld\n", "fmt_i64").unwrap();
                        let val_i64 = if val.get_type() == self.i64_type {
                            val
                        } else {
                            builder.build_int_s_extend(val, self.i64_type, "ext_i64").unwrap()
                        };
                        let _ = builder.build_call(printf_fn, &[fmt.as_pointer_value().into(), val_i64.into()], "printf_call");
                    }
                },
                Part::String(s) => {
                    let str_val = builder.build_global_string_ptr(&s, "str_lit").unwrap();
                    let _ = builder.build_call(puts_fn, &[str_val.as_pointer_value().into()], "puts_call");
                },
                Part::Integer(size, value, unsigned) => {
                    match size {
                        Size::S8 => int_values.push(self.i8_type.const_int(value, unsigned)),
                        Size::S16 => int_values.push(self.i16_type.const_int(value, unsigned)),
                        Size::S32 => int_values.push(self.i32_type.const_int(value, unsigned)),
                        Size::S64 => int_values.push(self.i64_type.const_int(value, unsigned)),
                        Size::S128 => int_values.push(self.i128_type.const_int(value, unsigned)),
                    }
                },
                Part::CalculateInteger(_size, mode) => {
                    let rhs = int_values.pop().unwrap();
                    let lhs = int_values.pop().unwrap();
                    match mode {
                        IntegerCalculateType::Add => int_values.push(builder.build_int_add(lhs, rhs, "add_tmp").unwrap()),
                        IntegerCalculateType::Sub => int_values.push(builder.build_int_sub(lhs, rhs, "sub_tmp").unwrap()),
                        IntegerCalculateType::Mul => int_values.push(builder.build_int_mul(lhs, rhs, "mul_tmp").unwrap()),
                        IntegerCalculateType::Div(unsigned) => {
                            if unsigned {
                                int_values.push(builder.build_int_unsigned_div(lhs, rhs, "div_tmp").unwrap())
                            } else {
                                int_values.push(builder.build_int_signed_div(lhs, rhs, "div_tmp").unwrap())
                            }
                        },
                        IntegerCalculateType::Rem(unsigned) => {
                            if unsigned {
                                int_values.push(builder.build_int_unsigned_rem(lhs, rhs, "rem_tmp").unwrap())
                            } else {
                                int_values.push(builder.build_int_signed_rem(lhs, rhs, "rem_tmp").unwrap())
                            }
                        }
                    }
                },
                Part::ComparisonInteger(mode, unsigned, _size) => {
                    let rhs = int_values.pop().unwrap();
                    let lhs = int_values.pop().unwrap();
                    match mode {
                        IntegerComparisonType::Equal => {
                            int_values.push(builder.build_int_compare(IntPredicate::EQ, lhs, rhs, "eq_tmp").unwrap());
                        },
                        IntegerComparisonType::NotEqual => {
                            int_values.push(builder.build_int_compare(IntPredicate::NE, lhs, rhs, "ne_tmp").unwrap());
                        },
                        IntegerComparisonType::Greater => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::UGT} else {IntPredicate::SGT}, lhs, rhs, "g_tmp").unwrap());
                        },
                        IntegerComparisonType::GreaterEqual => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::UGE} else {IntPredicate::SGE}, lhs, rhs, "ge_tmp").unwrap());
                        },
                        IntegerComparisonType::Less => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::ULT} else {IntPredicate::SLT}, lhs, rhs, "l_tmp").unwrap());
                        },
                        IntegerComparisonType::LessEqual => {
                            int_values.push(builder.build_int_compare(if unsigned {IntPredicate::ULE} else {IntPredicate::SLE}, lhs, rhs, "le_tmp").unwrap());
                        }
                    }
                },
                Part::ResizeInteger(current_size, to_size, z_or_s) => {
                    let value = int_values.pop().unwrap();
                    fn size_to_type<'a>(size: Size, rfc: &RuntimeFunctionCompiler<'a>) -> IntType<'a> {
                        match size {
                            Size::S8 => rfc.i8_type,
                            Size::S16 => rfc.i16_type,
                            Size::S32 => rfc.i32_type,
                            Size::S64 => rfc.i64_type,
                            Size::S128 => rfc.i128_type,
                        }
                    }
                    let to_type = size_to_type(to_size, self);
                    if current_size < to_size {
                        if z_or_s {
                            int_values.push(builder.build_int_z_extend(value, to_type, "tmp_rsz").unwrap())
                        } else {
                            int_values.push(builder.build_int_s_extend(value, to_type, "tmp_rsz").unwrap())
                        }
                    } else if current_size > to_size {
                        int_values.push(builder.build_int_truncate(value, to_type, "tmp_rsz").unwrap())
                    } else {
                        int_values.push(value);
                    }
                },
                Part::IntegerToDecimal(unsigned, mode) => {
                    let value = int_values.pop().unwrap();
                    let float_type = match mode { Size::S32 => self.f32_type, Size::S64 => self.f64_type, _ => unreachable!() };
                    float_values.push(
                        if unsigned { builder.build_unsigned_int_to_float(value, float_type, "tmp_i2f").unwrap() }
                        else { builder.build_signed_int_to_float(value, float_type, "tmp_i2f").unwrap() }
                    );
                },
                Part::ReturnInteger => {
                    if let Some(val) = int_values.pop() {
                        let ret_val = if val.get_type() == self.i64_type {
                            val
                        } else {
                            builder.build_int_s_extend(val, self.i64_type, "ret_ext").unwrap()
                        };
                        builder.build_return(Some(&ret_val)).ok();
                    } else {
                        builder.build_return(Some(&self.i64_type.const_int(0, false))).ok();
                    }
                },
                Part::Decimal(size, value) => {
                    match size {
                        Size::S32 => float_values.push(self.f32_type.const_float(value)),
                        Size::S64 => float_values.push(self.f64_type.const_float(value)),
                        _ => {}
                    }
                },
                Part::CalculateDecimal(_size, mode) => {
                    let rhs = float_values.pop().unwrap();
                    let lhs = float_values.pop().unwrap();
                    match mode {
                        DecimalCalculateType::Add => float_values.push(builder.build_float_add(lhs, rhs, "add_tmp").unwrap()),
                        DecimalCalculateType::Sub => float_values.push(builder.build_float_sub(lhs, rhs, "sub_tmp").unwrap()),
                        DecimalCalculateType::Mul => float_values.push(builder.build_float_mul(lhs, rhs, "mul_tmp").unwrap()),
                        DecimalCalculateType::Div => float_values.push(builder.build_float_div(lhs, rhs, "div_tmp").unwrap()),
                        DecimalCalculateType::Rem => float_values.push(builder.build_float_rem(lhs, rhs, "rem_tmp").unwrap()),
                    }
                },
                Part::DecimalNegative => {
                    let value = float_values.pop().unwrap();
                    float_values.push(builder.build_float_neg(value, "neg_tmp").unwrap());
                },
                Part::ComparisonDecimal(mode, ordered) => {
                    let rhs = float_values.pop().unwrap();
                    let lhs = float_values.pop().unwrap();
                    match mode {
                        DecimalComparisonType::Equal => int_values.push(builder.build_float_compare(if ordered { FloatPredicate::OEQ } else { FloatPredicate::UEQ }, lhs, rhs, "eq_tmp").unwrap()),
                        DecimalComparisonType::NotEqual => int_values.push(builder.build_float_compare(if ordered { FloatPredicate::ONE } else { FloatPredicate::UNE }, lhs, rhs, "ne_tmp").unwrap()),
                        DecimalComparisonType::Greater => int_values.push(builder.build_float_compare(if ordered { FloatPredicate::OGT } else { FloatPredicate::UGT }, lhs, rhs, "gt_tmp").unwrap()),
                        DecimalComparisonType::GreaterEqual => int_values.push(builder.build_float_compare(if ordered { FloatPredicate::OGE } else { FloatPredicate::UGE }, lhs, rhs, "ge_tmp").unwrap()),
                        DecimalComparisonType::Less => int_values.push(builder.build_float_compare(if ordered { FloatPredicate::OLT } else { FloatPredicate::ULT }, lhs, rhs, "lt_tmp").unwrap()),
                        DecimalComparisonType::LessEqual => int_values.push(builder.build_float_compare(if ordered { FloatPredicate::OLE } else { FloatPredicate::ULE }, lhs, rhs, "le_tmp").unwrap()),
                    }
                },
                Part::ResizeDecimal(extend) => {
                    let value = float_values.pop().unwrap();
                    if extend {
                        float_values.push(builder.build_float_ext(value, self.f64_type, "tmp_ext").unwrap());
                    } else {
                        float_values.push(builder.build_float_trunc(value, self.f32_type, "tmp_trc").unwrap());
                    }
                },
                Part::DecimalToInteger(unsigned, mode) => {
                    let value = float_values.pop().unwrap();
                    let int_type = match mode {
                        Size::S8 => self.i8_type,
                        Size::S16 => self.i16_type,
                        Size::S32 => self.i32_type,
                        Size::S64 => self.i64_type,
                        Size::S128 => self.i128_type,
                    };
                    int_values.push(
                        if unsigned { builder.build_float_to_unsigned_int(value, int_type, "tmp_i2f").unwrap() }
                        else { builder.build_float_to_signed_int(value, int_type, "tmp_i2f").unwrap() }
                    );
                },
                Part::BinaryOperation(mode) => {
                    let rhs = int_values.pop().unwrap();
                    let lhs = int_values.pop().unwrap();
                    match mode {
                        BinaryOperationType::LShift => int_values.push(builder.build_left_shift(lhs, rhs, "ls").unwrap()),
                        BinaryOperationType::RShift(sg) => int_values.push(builder.build_right_shift(lhs, rhs, sg, "ls").unwrap()),
                        BinaryOperationType::And => int_values.push(builder.build_and(lhs, rhs, "and_tmp").unwrap()),
                        BinaryOperationType::Or => int_values.push(builder.build_or(lhs, rhs, "or_tmp").unwrap()),
                        BinaryOperationType::Xor => int_values.push(builder.build_xor(lhs, rhs, "xor_tmp").unwrap()),
                    }
                },
                Part::Bool(value) => {
                    int_values.push(self.bool_type.const_int(if value {1} else {0}, false));
                },
                Part::Pop => {
                    let _ = int_values.pop();
                },
                Part::Dup => {
                    if let Some(top) = int_values.last().cloned() {
                        int_values.push(top);
                    }
                },
                Part::Swap => {
                    let len = int_values.len();
                    if len >= 2 {
                        int_values.swap(len - 1, len - 2);
                    }
                },
                Part::RotateThree => {
                    let len = int_values.len();
                    if len >= 3 {
                        let c = int_values.pop().unwrap();
                        let b = int_values.pop().unwrap();
                        let a = int_values.pop().unwrap();
                        int_values.push(b);
                        int_values.push(c);
                        int_values.push(a);
                    }
                },
                Part::PushNull => {
                    int_values.push(self.i64_type.const_int(0, false));
                },
                Part::Nop => {},
                Part::Label(name) => {
                    if let Some(target_bb) = label_blocks.get(&name) {
                        if let Some(current_bb) = builder.get_insert_block() {
                            if current_bb.get_terminator().is_none() {
                                builder.build_unconditional_branch(*target_bb).ok();
                            }
                        }
                        builder.position_at_end(*target_bb);
                    }
                },
                Part::Jump(name) => {
                    if let Some(target_bb) = label_blocks.get(&name) {
                        builder.build_unconditional_branch(*target_bb).ok();
                        let dead_bb = self.context.append_basic_block(function, "after_jump");
                        builder.position_at_end(dead_bb);
                    }
                },
                Part::JumpIfTrue(name) => {
                    if let Some(val) = int_values.pop() {
                        if let Some(target_bb) = label_blocks.get(&name) {
                            let cond = if val.get_type() == self.bool_type {
                                val
                            } else {
                                let zero = val.get_type().const_int(0, false);
                                builder.build_int_compare(IntPredicate::NE, val, zero, "tobool").unwrap()
                            };
                            let fallthrough_bb = self.context.append_basic_block(function, "fallthrough");
                            builder.build_conditional_branch(cond, *target_bb, fallthrough_bb).ok();
                            builder.position_at_end(fallthrough_bb);
                        }
                    }
                },
                Part::JumpIfFalse(name) => {
                    if let Some(val) = int_values.pop() {
                        if let Some(target_bb) = label_blocks.get(&name) {
                            let is_true = if val.get_type() == self.bool_type {
                                val
                            } else {
                                let zero = val.get_type().const_int(0, false);
                                builder.build_int_compare(IntPredicate::NE, val, zero, "to_bool").unwrap()
                            };
                            let fallthrough_bb = self.context.append_basic_block(function, "fallthrough");
                            builder.build_conditional_branch(is_true, fallthrough_bb, *target_bb).ok();
                            builder.position_at_end(fallthrough_bb);
                        }
                    }
                },
                Part::Halt => break,
                Part::BitwiseNot(_) => {
                    if let Some(val) = int_values.pop() {
                        int_values.push(builder.build_not(val, "not_tmp").unwrap());
                    }
                },
                Part::ReturnVoid => {
                    if let Some(current_bb) = builder.get_insert_block() {
                        if current_bb.get_terminator().is_none() {
                            builder.build_return(Some(&self.i64_type.const_int(0, false))).ok();
                        }
                    }
                },
                Part::PushLocal(name) => {
                    let alloca = locals_map.entry(name.clone()).or_insert_with(|| {
                        builder.build_alloca(self.i64_type, &name).unwrap()
                    });
                    let loaded = builder.build_load(self.i64_type, *alloca, "loc_load").unwrap();
                    int_values.push(loaded.into_int_value());
                },
                Part::StoreLocal(name) => {
                    if let Some(val) = int_values.pop() {
                        let alloca = locals_map.entry(name).or_insert_with(|| {
                            builder.build_alloca(self.i64_type, "loc").unwrap()
                        });
                        let val_i64 = if val.get_type() == self.i64_type {
                            val
                        } else {
                            builder.build_int_s_extend(val, self.i64_type, "loc_ext").unwrap()
                        };
                        builder.build_store(*alloca, val_i64).unwrap();
                    }
                },
                Part::AllocStackStruct(_name) | Part::AllocStruct(_name) => {
                    let alloca = builder.build_alloca(self.i64_type, "struct_alloc").unwrap();
                    let ptr_int = builder.build_ptr_to_int(alloca, self.i64_type, "ptr_int").unwrap();
                    int_values.push(ptr_int);
                },
                Part::GetStructField { offset, size, .. } => {
                    let base = int_values.pop().unwrap_or_else(|| self.i64_type.const_int(0, false));
                    let base_ptr = builder.build_int_to_ptr(base, self.context.ptr_type(AddressSpace::from(0)), "base_ptr").unwrap();
                    let offset_val = self.i64_type.const_int(offset as u64, false);
                    let field_ptr = unsafe { builder.build_gep(self.i8_type, base_ptr, &[offset_val], "gep").unwrap() };
                    let ty = match size {
                        Size::S8 => self.i8_type,
                        Size::S16 => self.i16_type,
                        Size::S32 => self.i32_type,
                        _ => self.i64_type,
                    };
                    let loaded = builder.build_load(ty, field_ptr, "fld").unwrap();
                    int_values.push(loaded.into_int_value());
                },
                Part::SetStructField { offset, .. } => {
                    if let Some(val) = int_values.pop() {
                        if let Some(base) = int_values.last().cloned() {
                            let base_ptr = builder.build_int_to_ptr(base, self.context.ptr_type(AddressSpace::from(0)), "base_ptr").unwrap();
                            let offset_val = self.i64_type.const_int(offset as u64, false);
                            let field_ptr = unsafe { builder.build_gep(self.i8_type, base_ptr, &[offset_val], "gep").unwrap() };
                            builder.build_store(field_ptr, val).unwrap();
                        }
                    }
                },
                Part::GetStructFieldPtr { offset, .. } => {
                    if let Some(base) = int_values.pop() {
                        let offset_val = self.i64_type.const_int(offset as u64, false);
                        int_values.push(builder.build_int_add(base, offset_val, "ptr_add").unwrap());
                    }
                },
                Part::UnaryMath(_size, math_type) => {
                    match math_type {
                        UnaryMathType::Abs => {
                            if let Some(val) = int_values.pop() {
                                let zero = val.get_type().const_int(0, false);
                                let neg = builder.build_int_neg(val, "neg").unwrap();
                                let is_neg = builder.build_int_compare(IntPredicate::SLT, val, zero, "is_neg").unwrap();
                                let abs = builder.build_select(is_neg, neg, val, "abs").unwrap();
                                int_values.push(abs.into_int_value());
                            }
                        },
                        UnaryMathType::Negate => {
                            if let Some(val) = int_values.pop() {
                                int_values.push(builder.build_int_neg(val, "neg").unwrap());
                            }
                        },
                        UnaryMathType::Increment => {
                            if let Some(val) = int_values.pop() {
                                let one = val.get_type().const_int(1, false);
                                int_values.push(builder.build_int_add(val, one, "inc").unwrap());
                            }
                        },
                        UnaryMathType::Decrement => {
                            if let Some(val) = int_values.pop() {
                                let one = val.get_type().const_int(1, false);
                                int_values.push(builder.build_int_sub(val, one, "dec").unwrap());
                            }
                        },
                        _ => {}
                    }
                },
                Part::FusedMultiplyAdd(_) => {
                    if float_values.len() >= 3 {
                        let c = float_values.pop().unwrap();
                        let b = float_values.pop().unwrap();
                        let a = float_values.pop().unwrap();
                        let mul = builder.build_float_mul(a, b, "fma_mul").unwrap();
                        float_values.push(builder.build_float_add(mul, c, "fma_add").unwrap());
                    }
                },
                _ => {}
            }
        }

        if let Some(current_bb) = builder.get_insert_block() {
            if current_bb.get_terminator().is_none() {
                builder.build_return(Some(&self.i64_type.const_int(0, false))).ok();
            }
        }

        Some(function)
    }

    pub fn jit_compile<'a>(
        &self,
        paper: Paper,
        ee: &'a ExecutionEngine<'ctx>
    ) -> Option<JitFunction<'a, Func>> {
        let _ = self.compile_paper_function(&paper)?;
        unsafe {
            ee.get_function::<Func>(paper.name.as_str()).ok()
        }
    }

    pub fn ivm_compile(
        &self,
        paper: Paper,
    ) -> Option<Chunk> {
        let mut chunk = Chunk::new();
        let mut local_slots: std::collections::HashMap<String, u8> = std::collections::HashMap::new();
        local_slots.insert("global_id".to_string(), 0);
        local_slots.insert("thread_idx".to_string(), 1);
        local_slots.insert("block_idx".to_string(), 2);
        let mut next_slot: u8 = 3;
        let mut labels: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut jumps_to_patch: Vec<(usize, String, bool)> = Vec::new();

        for part in paper.parts {
            match part {
                Part::Print => {
                    chunk.write(OpCode::PrintTopOfStack);
                },
                Part::String(s) => {
                    let const_idx = chunk.add_constant(Value::Str(std::rc::Rc::new(s)));
                    chunk.write(OpCode::PushConstant8);
                    chunk.write(const_idx);
                },
                Part::Integer(size, value, _unsigned) => {
                    match size {
                        Size::S8 => {
                            chunk.write(OpCode::LoadImmediateI8);
                            chunk.write(value as u8);
                        },
                        Size::S16 => {
                            chunk.write(OpCode::LoadImmediateI16);
                            chunk.write(value as u16);
                        },
                        Size::S32 => {
                            chunk.write(OpCode::LoadImmediateI32);
                            chunk.write(value as i32);
                        },
                        Size::S64 => {
                            chunk.write(OpCode::LoadImmediateI64);
                            chunk.write(value as i64);
                        },
                        Size::S128 => {


                        }
                    }
                },
                Part::CalculateInteger(size, mode) => {
                    match size {
                        Size::S8 => match mode {
                            IntegerCalculateType::Add => chunk.write(OpCode::AddInt8),
                            IntegerCalculateType::Sub => chunk.write(OpCode::SubtractInt8),
                            IntegerCalculateType::Mul => chunk.write(OpCode::MultiplyInt8),
                            IntegerCalculateType::Div(_) => chunk.write(OpCode::DivideInt8),
                            IntegerCalculateType::Rem(_) => chunk.write(OpCode::ModuloInt32),
                        },
                        Size::S16 => match mode {
                            IntegerCalculateType::Add => chunk.write(OpCode::AddInt16),
                            IntegerCalculateType::Sub => chunk.write(OpCode::SubtractInt16),
                            IntegerCalculateType::Mul => chunk.write(OpCode::MultiplyInt16),
                            IntegerCalculateType::Div(_) => chunk.write(OpCode::DivideInt16),
                            IntegerCalculateType::Rem(_) => chunk.write(OpCode::ModuloInt32),
                        },
                        Size::S32 => match mode {
                            IntegerCalculateType::Add => chunk.write(OpCode::AddInt32),
                            IntegerCalculateType::Sub => chunk.write(OpCode::SubtractInt32),
                            IntegerCalculateType::Mul => chunk.write(OpCode::MultiplyInt32),
                            IntegerCalculateType::Div(unsigned) => {
                                chunk.write(if unsigned { OpCode::DivideUnsigned32 } else { OpCode::DivideInt32 });
                            }
                            IntegerCalculateType::Rem(_) => chunk.write(OpCode::ModuloInt32),
                        },
                        Size::S64 => match mode {
                            IntegerCalculateType::Add => chunk.write(OpCode::AddInt64),
                            IntegerCalculateType::Sub => chunk.write(OpCode::SubtractInt64),
                            IntegerCalculateType::Mul => chunk.write(OpCode::MultiplyInt64),
                            IntegerCalculateType::Div(unsigned) => {
                                chunk.write(if unsigned { OpCode::DivideUnsigned64 } else { OpCode::DivideInt64 });
                            }
                            IntegerCalculateType::Rem(_) => chunk.write(OpCode::ModuloInt64),
                        },
                        Size::S128 => {}
                    }
                },
                Part::ComparisonInteger(mode, _unsigned, size) => {
                    match mode {
                        IntegerComparisonType::Equal => {
                            chunk.write(if size {OpCode::EqualInt64} else {OpCode::EqualInt32})
                        },
                        IntegerComparisonType::NotEqual => {
                            chunk.write(if size {OpCode::NotEqualInt64} else {OpCode::NotEqualInt32})
                        },
                        IntegerComparisonType::Greater => {
                            chunk.write(if size {OpCode::GreaterThanInt64} else {OpCode::GreaterThanInt32})
                        },
                        IntegerComparisonType::GreaterEqual => {
                            chunk.write(if size {OpCode::GreaterOrEqualInt64} else {OpCode::GreaterOrEqualInt32})
                        },
                        IntegerComparisonType::Less => {
                            chunk.write(if size {OpCode::LessThanInt64} else {OpCode::LessThanInt32})
                        },
                        IntegerComparisonType::LessEqual => {
                            chunk.write(if size {OpCode::LessOrEqualInt64} else {OpCode::LessOrEqualInt32})
                        }
                    }
                },
                Part::ResizeInteger(_current_size, _to_size, _z_or_s) => {


                },
                Part::ReturnInteger => {
                    chunk.write(OpCode::ReturnFromFunction);
                },
                Part::Decimal(size, value) => {
                    match size {
                        Size::S32 => {
                            chunk.write(OpCode::LoadImmediateF32);
                            chunk.write(value as f32);
                        }
                        Size::S64 => {
                            chunk.write(OpCode::LoadImmediateF64);
                            chunk.write(value);
                        }
                        _ => {}
                    }
                }
                Part::CalculateDecimal(size, mode) => {
                    match size {
                        Size::S32 => {
                            match mode {
                                DecimalCalculateType::Add => {
                                    chunk.write(OpCode::AddFloat32);
                                },
                                DecimalCalculateType::Sub => {
                                    chunk.write(OpCode::SubtractFloat32);
                                },
                                DecimalCalculateType::Mul => {
                                    chunk.write(OpCode::MultiplyFloat32);
                                },
                                DecimalCalculateType::Div => {
                                    chunk.write(OpCode::DivideFloat32);
                                },
                                DecimalCalculateType::Rem => {
                                    chunk.write(OpCode::ModuloFloat32);
                                }
                            }
                        }
                        Size::S64 => {
                            match mode {
                                DecimalCalculateType::Add => {
                                    chunk.write(OpCode::AddFloat64);
                                },
                                DecimalCalculateType::Sub => {
                                    chunk.write(OpCode::SubtractFloat64);
                                },
                                DecimalCalculateType::Mul => {
                                    chunk.write(OpCode::MultiplyFloat64);
                                },
                                DecimalCalculateType::Div => {
                                    chunk.write(OpCode::DivideFloat64);
                                },
                                DecimalCalculateType::Rem => {
                                    chunk.write(OpCode::ModuloFloat64);
                                }
                            }
                        }
                        _ => {}
                    }
                },
                Part::Pop => chunk.write(OpCode::PopStack),
                Part::Dup => chunk.write(OpCode::DuplicateTop),
                Part::Swap => chunk.write(OpCode::SwapTopTwo),
                Part::RotateThree => chunk.write(OpCode::RotateTopThree),
                Part::PushNull => chunk.write(OpCode::PushNull),
                Part::Bool(b) => chunk.write(if b { OpCode::PushTrue } else { OpCode::PushFalse }),
                Part::Nop => chunk.write(OpCode::NoOperation),
                Part::Halt => chunk.write(OpCode::Halt),
                Part::ReturnVoid => chunk.write(OpCode::ReturnFromFunction),
                Part::GetArrayLength => chunk.write(OpCode::GetArrayLength),
                Part::GetArrayElement => chunk.write(OpCode::GetArrayIndexInt32),
                Part::SetArrayElement => chunk.write(OpCode::SetArrayIndexInt32),
                Part::ResizeArray => chunk.write(OpCode::ResizeArray),
                Part::MapContainsKey => chunk.write(OpCode::MapContainsKey),
                Part::MapRemoveKey => chunk.write(OpCode::MapRemoveKey),
                Part::MapGetOrDefault => chunk.write(OpCode::MapGetOrDefaultValue),
                Part::AllocateSlice => chunk.write(OpCode::NoOperation),
                Part::EnterMonitor => chunk.write(OpCode::EnterMonitor),
                Part::ExitMonitor => chunk.write(OpCode::ExitMonitor),
                Part::YieldThread => chunk.write(OpCode::YieldCurrentThread),
                Part::Throw => chunk.write(OpCode::ThrowException),
                Part::EndTry => chunk.write(OpCode::EndTryBlock),
                Part::UnaryMath(size, UnaryMathType::Abs) => match size {
                    Size::S32 => chunk.write(OpCode::AbsoluteInt32),
                    Size::S64 => chunk.write(OpCode::AbsoluteInt64),
                    _ => {}
                },
                Part::UnaryMath(size, UnaryMathType::Sqrt) => match size {
                    Size::S32 => chunk.write(OpCode::SquareRootFloat32),
                    Size::S64 => chunk.write(OpCode::SquareRootFloat64),
                    _ => {}
                },
                Part::UnaryMath(_size, UnaryMathType::Floor) => chunk.write(OpCode::FloorFloat32),
                Part::UnaryMath(_size, UnaryMathType::Ceil) => chunk.write(OpCode::CeilFloat32),
                Part::UnaryMath(_size, UnaryMathType::Round) => chunk.write(OpCode::RoundFloat32),
                Part::UnaryMath(_size, UnaryMathType::Truncate) => chunk.write(OpCode::TruncateFloat32),
                Part::UnaryMath(size, UnaryMathType::Increment) => match size {
                    Size::S32 => chunk.write(OpCode::IncrementInt32),
                    Size::S64 => chunk.write(OpCode::IncrementInt64),
                    _ => {}
                },
                Part::UnaryMath(size, UnaryMathType::Decrement) => match size {
                    Size::S32 => chunk.write(OpCode::DecrementInt32),
                    Size::S64 => chunk.write(OpCode::DecrementInt64),
                    _ => {}
                },
                Part::UnaryMath(size, UnaryMathType::Negate) => match size {
                    Size::S32 => chunk.write(OpCode::NegateInt32),
                    Size::S64 => chunk.write(OpCode::NegateInt64),
                    _ => {}
                },
                Part::FusedMultiplyAdd(size) => match size {
                    Size::S32 => chunk.write(OpCode::FusedMultiplyAddFloat32),
                    Size::S64 => chunk.write(OpCode::FusedMultiplyAddFloat64),
                    _ => {}
                },
                Part::AtomicAdd(_) => chunk.write(OpCode::AtomicAddInt32),
                Part::AtomicSub(_) => chunk.write(OpCode::AtomicSubtractInt32),
                Part::AtomicCas(_) => chunk.write(OpCode::AtomicCompareAndSwapInt32),
                Part::SpawnDetached(name) => {
                    chunk.write(OpCode::SpawnDetached);
                    let const_idx = chunk.add_constant(Value::Str(std::rc::Rc::new(name)));
                    chunk.write(const_idx as u16);
                }
                Part::AllocStackStruct(_) => {
                    chunk.write(OpCode::AllocStackStruct);
                    chunk.write(64u16);
                }
                Part::AllocStruct(_) => {
                    chunk.write(OpCode::AllocStruct);
                    chunk.write(64u16);
                }
                Part::GetStructField { offset, size, .. } => {
                    chunk.write(OpCode::GetStructField);
                    chunk.write(offset as u16);
                    chunk.write(size as u8);
                }
                Part::SetStructField { offset, size, .. } => {
                    chunk.write(OpCode::SetStructField);
                    chunk.write(offset as u16);
                    chunk.write(size as u8);
                }
                Part::GetStructFieldPtr { offset, .. } => {
                    chunk.write(OpCode::GetStructFieldPtr);
                    chunk.write(offset as u16);
                }
                Part::CallMethod { .. } => {
                    chunk.write(OpCode::CallMethod);
                    chunk.write(0u16);
                }
                Part::AllocRaw { size, .. } => {
                    chunk.write(OpCode::AllocRaw);
                    chunk.write(size as u16);
                }
                Part::FreeRaw => chunk.write(OpCode::FreeRaw),
                Part::LoadPtr(size) => {
                    chunk.write(OpCode::LoadPtr);
                    chunk.write(size as u8);
                }
                Part::StorePtr(size) => {
                    chunk.write(OpCode::StorePtr);
                    chunk.write(size as u8);
                }
                Part::OffsetPtr(offset) => {
                    chunk.write(OpCode::OffsetPtr);
                    chunk.write(offset as i16);
                }
                Part::KernelLaunch { .. } => {
                    chunk.write(OpCode::KernelLaunch);
                    chunk.write(0u16);
                }
                Part::DeviceMemoryAlloc(_, size) => {
                    chunk.write(OpCode::DeviceMemoryAlloc);
                    chunk.write(size as u16);
                }
                Part::CopyToDevice(_) => {
                    chunk.write(OpCode::CopyToDevice);
                    chunk.write(0u16);
                }
                Part::CopyFromDevice(_) => {
                    chunk.write(OpCode::CopyFromDevice);
                    chunk.write(0u16);
                }
                Part::BitwiseNot(size) => match size {
                    Size::S32 => chunk.write(OpCode::BitwiseNotInt32),
                    _ => chunk.write(OpCode::BitwiseNotInt64),
                },
                Part::Label(name) => {
                    labels.insert(name, chunk.code.len());
                }
                Part::PushLocal(name) => {
                    let slot = *local_slots.entry(name).or_insert_with(|| {
                        let s = next_slot;
                        next_slot += 1;
                        s
                    });
                    chunk.write(OpCode::GetLocalVariable8);
                    chunk.write(slot);
                }
                Part::StoreLocal(name) => {
                    let slot = *local_slots.entry(name).or_insert_with(|| {
                        let s = next_slot;
                        next_slot += 1;
                        s
                    });
                    chunk.write(OpCode::SetLocalVariable8);
                    chunk.write(slot);
                }
                Part::PushGlobal(_) => chunk.write(OpCode::GetGlobalVariable8),
                Part::StoreGlobal(_) => chunk.write(OpCode::SetGlobalVariable8),
                Part::Jump(target) => {
                    if let Some(_target_pos) = labels.get(&target) {
                        chunk.write(OpCode::LoopJump);
                        let patch_pos = chunk.code.len();
                        chunk.write(0u16);
                        jumps_to_patch.push((patch_pos, target, true));
                    } else {
                        chunk.write(OpCode::UnconditionalJump);
                        let patch_pos = chunk.code.len();
                        chunk.write(0u16);
                        jumps_to_patch.push((patch_pos, target, false));
                    }
                }
                Part::JumpIfTrue(target) => {
                    chunk.write(OpCode::JumpIfTrue);
                    let patch_pos = chunk.code.len();
                    chunk.write(0u16);
                    jumps_to_patch.push((patch_pos, target, false));
                }
                Part::JumpIfFalse(target) => {
                    chunk.write(OpCode::JumpIfFalse);
                    let patch_pos = chunk.code.len();
                    chunk.write(0u16);
                    jumps_to_patch.push((patch_pos, target, false));
                }
                Part::JumpIfNull(_) => chunk.write(OpCode::JumpIfNull),
                Part::JumpIfNonNull(_) => chunk.write(OpCode::JumpIfNonNull),
                Part::Call(..) => chunk.write(OpCode::CallFunction),
                Part::TailCall(..) => chunk.write(OpCode::TailCallFunction),
                Part::TableSwitch { .. } => chunk.write(OpCode::TableSwitch),
                Part::CreateArray { .. } => chunk.write(OpCode::CreateNewArray8),
                Part::CreateMap { .. } => chunk.write(OpCode::CreateNewMap8),
                Part::Try(_) => chunk.write(OpCode::BeginTryBlock),
                Part::CallWithCache { .. } => chunk.write(OpCode::CallWithInlineCache),
                Part::GetPropertyWithCache(_) => chunk.write(OpCode::GetPropertyWithInlineCache),
                Part::SetPropertyWithCache(_) => chunk.write(OpCode::SetPropertyWithInlineCache),
                Part::MegamorphicCall { .. } => chunk.write(OpCode::MegamorphicMethodCall),
                _ => {}
            }
        }


        for (patch_pos, target_name, is_backward) in jumps_to_patch {
            if is_backward {
                let target_pos = labels.get(&target_name).copied().unwrap_or(0);
                let offset = (patch_pos + 2).saturating_sub(target_pos) as u16;
                let bytes = offset.to_be_bytes();
                chunk.code[patch_pos] = bytes[0];
                chunk.code[patch_pos + 1] = bytes[1];
            } else {
                let target_pos = labels.get(&target_name).copied().unwrap_or(chunk.code.len());
                let offset = target_pos.saturating_sub(patch_pos + 2) as u16;
                let bytes = offset.to_be_bytes();
                chunk.code[patch_pos] = bytes[0];
                chunk.code[patch_pos + 1] = bytes[1];
            }
        }

        Some(chunk)
    }
}

#[no_mangle]
#[allow(unused)]
pub extern "C" fn iris_print_logic(engine: *mut IrisEngine) {
    unsafe {
        let engine_ref = &mut *engine;

        if !engine_ref.stack.is_empty() {
            let value = engine_ref.stack.pop().unwrap();
            match value {
                Value::Str(ref s) => println!("{}", s),
                Value::I32(v) => println!("{}", v),
                Value::I64(v) => println!("{}", v),
                _ => println!("{:?}", value),
            }
        }
    }
}
#[no_mangle]
pub extern "C" fn iris_push_str(engine: *mut IrisEngine, str_ptr: *const u8, len: usize) {
    unsafe {
        let slice = std::slice::from_raw_parts(str_ptr, len);
        if let Ok(s) = std::str::from_utf8(slice) {
            (*engine).stack.push(Value::Str(std::rc::Rc::new(s.to_string())));
        }
    }
}
#[no_mangle]
pub extern "C" fn iris_push_i32_logic(engine: *mut IrisEngine, value: i32) {
    unsafe {
        let engine_ref = &mut *engine;

        engine_ref.stack.push(Value::I32(value));
    }
}
#[no_mangle]
pub extern "C" fn iris_push_i64_logic(engine: *mut IrisEngine, value: i64) {
    unsafe {
        let engine_ref = &mut *engine;

        engine_ref.stack.push(Value::I64(value));
    }
}
#[no_mangle]
pub extern "C" fn iris_get_global_id(engine: *mut IrisEngine) -> i64 {
    unsafe { (*engine).global_id as i64 }
}
#[no_mangle]
pub extern "C" fn iris_get_thread_idx(engine: *mut IrisEngine) -> i64 {
    unsafe { (*engine).thread_idx as i64 }
}
#[no_mangle]
pub extern "C" fn iris_get_block_idx(engine: *mut IrisEngine) -> i64 {
    unsafe { (*engine).block_idx as i64 }
}
#[no_mangle]
pub extern "C" fn iris_spawn_detached(name_ptr: *const u8, name_len: usize) {
    let slice = unsafe { std::slice::from_raw_parts(name_ptr, name_len) };
    if let Ok(name) = std::str::from_utf8(slice) {
        if let Some(paper) = crate::vm::pool::get_global_paper(name) {
            crate::vm::pool::IrisThreadPool::global().spawn(move || {
                let mut vm = IrisEngine::new();
                let _ = paper.run_vm(&mut vm);
            });
        }
    }
}
