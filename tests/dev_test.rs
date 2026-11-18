use inkwell::context::Context;
use iris::builder::compiler::{RuntimeFunctionCompiler};
use iris::builder::paper::Paper;
use iris::builder::part::{IntegerCalculateType, Part, Size};
use iris::vm::{
    engine::IrisEngine,
};
#[test]
fn test_invoke_method() {

    let mut vm = IrisEngine::new();

    let paper = Paper {
        name: "Print".to_string(),
        entry_point: "main".to_string(),
        parts: vec![
            Part::Integer(Size::S32, 10, false),
            Part::Integer(Size::S32, 10, false),
            Part::CalculateInteger(Size::S32, IntegerCalculateType::Add),
            Part::Integer(Size::S32, 2, false),
            Part::CalculateInteger(Size::S32, IntegerCalculateType::Sub),
            Part::ReturnInteger, Part::Print
        ],
        locals: vec![],
        stack_size: 1024,
    };
    let context = Context::create();
    let module = context.create_module("print");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc.module.create_jit_execution_engine(inkwell::OptimizationLevel::None).ok().unwrap();

    rfc.setup_ee(&ee);
    
    {
        let compiled_print = rfc.jit_compile(paper, &ee);

        unsafe {
            compiled_print.unwrap().call(&mut vm);
        }
    }
    #[cfg(not(all()))]
    {
        let chunk = rfc.ivm_compile(paper).unwrap();
        let function = Rc::new(Function::new_bytecode("test_func".to_string(), 0, chunk.code, chunk.constants));
        let _ = vm.push_frame(function, 0);
        let _ = vm.run();
    }
}

#[test]
fn test_paper_full_opcodes() {
    use iris::builder::part::UnaryMathType;

    let paper = Paper::new("FullOpCodesTest", vec![
        Part::Nop,
        Part::PushNull,
        Part::Bool(true),
        Part::Pop,
        Part::Integer(Size::S32, 42, false),
        Part::Dup,
        Part::Swap,
        Part::RotateThree,
        Part::UnaryMath(Size::S32, UnaryMathType::Increment),
        Part::UnaryMath(Size::S32, UnaryMathType::Decrement),
        Part::UnaryMath(Size::S32, UnaryMathType::Abs),
        Part::GetArrayLength,
        Part::GetArrayElement,
        Part::SetArrayElement,
        Part::MapContainsKey,
        Part::MapRemoveKey,
        Part::EnterMonitor,
        Part::ExitMonitor,
        Part::YieldThread,
        Part::ReturnVoid,
    ])
    .with_entry_point("main")
    .with_stack_size(2048);

    let context = Context::create();
    let module = context.create_module("test_full");
    let rfc = RuntimeFunctionCompiler::new(&context, module);

    let chunk = rfc.ivm_compile(paper.clone()).expect("Failed to compile bytecode chunk");
    assert!(!chunk.code.is_empty());
}

#[test]
fn test_struct_and_impl() {
    use iris::builder::part::{FieldType, StructField};

    let point_struct = Part::DefineStruct {
        name: "Point".to_string(),
        size: 16,
        align: 8,
        fields: vec![
            StructField::new("x", FieldType::Integer(Size::S64), 0),
            StructField::new("y", FieldType::Integer(Size::S64), 8),
        ],
    };

    let paper = Paper::new("PointOperations", vec![
        point_struct,
        
        Part::AllocStackStruct("Point".to_string()),
        Part::Dup,
        
        Part::Integer(Size::S64, 10, false),
        Part::SetStructField {
            struct_name: "Point".to_string(),
            field_index: 0,
            offset: 0,
            size: Size::S64,
        },
        
        Part::Dup,
        Part::Integer(Size::S64, 20, false),
        Part::SetStructField {
            struct_name: "Point".to_string(),
            field_index: 1,
            offset: 8,
            size: Size::S64,
        },
        
        Part::CallMethod {
            struct_name: "Point".to_string(),
            method_name: "length".to_string(),
            args: vec![],
        },
        Part::ReturnInteger,
    ]);

    assert_eq!(paper.name, "PointOperations");
    assert_eq!(paper.parts.len(), 10);
}
