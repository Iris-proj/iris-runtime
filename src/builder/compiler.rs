use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::types::{FloatType, FunctionType, IntType, PointerType, VoidType};
use inkwell::AddressSpace;
use crate::builder::paper::Paper;
use crate::builder::part::Part;
use crate::vm::engine::IrisEngine;

pub type Func = unsafe extern "C" fn(iris_engine: *mut IrisEngine) -> u8;

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
    u8_type: IntType<'ctx>,
    u16_type: IntType<'ctx>,
    u32_type: IntType<'ctx>,
    u64_type: IntType<'ctx>,
    u128_type: IntType<'ctx>,
    f32_type: FloatType<'ctx>,
    f64_type: FloatType<'ctx>,
}

impl<'ctx> RuntimeFunctionCompiler<'ctx> {
    pub fn new(context: &'ctx Context, module: Module<'ctx>) -> Self {
        let i8_type = context.i8_type();
        let vm_type = context.ptr_type(AddressSpace::from(0));

        RuntimeFunctionCompiler {
            context,
            module,
            vm_type,
            function_type: i8_type.fn_type(&[vm_type.into()], false),
            null_type: context.void_type(),
            bool_type: context.bool_type(),
            i8_type,
            i16_type: context.i16_type(),
            i32_type: context.i32_type(),
            i64_type: context.i64_type(),
            i128_type: context.i128_type(),
            u8_type: context.i8_type(),
            u16_type: context.i16_type(),
            u32_type: context.i32_type(),
            u64_type: context.i64_type(),
            u128_type: context.i128_type(),
            f32_type: context.f32_type(),
            f64_type: context.f64_type(),
        }
    }

    pub fn jit_compile<'a>(
        &self,
        paper: Paper,
        ee: &'a ExecutionEngine<'ctx>
    ) -> Option<JitFunction<'a, Func>> {
        let function = self.module.add_function(paper.name.as_str(), self.function_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        let builder = self.context.create_builder();
        builder.position_at_end(basic_block);

        let engine_ptr = function.get_first_param()?.into_pointer_value();

        for part in paper.parts {
            match part {
                Part::Print => {
                    let print_fn_type = self.null_type.fn_type(&[self.vm_type.into()], false);
                    let print_fn = self.module.get_function("iris_print_logic")
                        .unwrap_or_else(|| self.module.add_function("iris_print_logic", print_fn_type, None));

                    let _ = builder.build_call(print_fn, &[engine_ptr.into()], "print_call");
                }
            }
        }

        builder.build_return(Some(&self.u8_type.const_int(0, false))).ok();

        unsafe {
            ee.get_function::<Func>(paper.name.as_str()).ok()
        }
    }
}

#[no_mangle]
pub extern "C" fn iris_print_logic(engine: *mut IrisEngine) {
    unsafe {
        if let Some(engine_ref) = engine.as_mut() {
            println!("{:?}", engine_ref.stack.pop());
        }
    }
}