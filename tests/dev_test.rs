use std::rc::Rc;
use inkwell::context::Context;
use iris::builder::compiler::{RuntimeFunctionCompiler};
use iris::builder::paper::Paper;
use iris::builder::part::Part;
use iris::vm::{
    chunk::ChunkWriter,
    function::Function,
    value::Value,
    engine::IrisEngine,
};
use iris::vm::chunk::Chunk;
use iris::vm::opcode::OpCode;

#[test]
fn test_invoke_method() {
    let mut chunk = Chunk::new();

    let hello_world = chunk.add_constant(Value::Str(Rc::new("Hello World".to_string())));

    chunk.write(OpCode::PushConstant8);
    chunk.write(hello_world);

    let mut vm = IrisEngine::new();
    let function = Rc::new(Function::new_bytecode("test_func".to_string(), 0, chunk.code, chunk.constants));
        let _ = vm.push_frame(function, 0);
    let _ = vm.run();

    let paper = Paper{
        name: "Print".to_string(),
        parts: vec![Part::Print]
    };
    let context = Context::create();
    let module = context.create_module("print");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc.module.create_jit_execution_engine(inkwell::OptimizationLevel::None).ok().unwrap();
    let compiled_print = rfc.jit_compile(paper, &ee);

    unsafe {
        compiled_print.unwrap().call(&mut vm);
    }
}
