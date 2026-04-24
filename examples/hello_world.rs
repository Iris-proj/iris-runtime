use std::rc::Rc;
use iris::vm::{
    value::Value,
    engine::IrisEngine,
};
use iris::vm::chunk::{Chunk, ChunkWriter};
use iris::vm::function::Function;
use iris::vm::opcode::OpCode;

#[test]
fn test_invoke_method() {
    let mut chunk = Chunk::new();

    let hello_world = chunk.add_constant(Value::Str(Rc::new("Hello World".to_string())));

    chunk.write(OpCode::PushConstant8);
    chunk.write(hello_world);
    chunk.write(OpCode::PrintTopOfStack);

    let mut vm = IrisEngine::new();
    let function = Rc::new(Function::new_bytecode("test_func".to_string(), 0, chunk.code, chunk.constants));
    let _ = vm.push_frame(function, 0);
    let _ = vm.run();
}
