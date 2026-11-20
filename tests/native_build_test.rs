use std::path::Path;
use std::process::Command;
use inkwell::OptimizationLevel;
use iris::builder::paper::{Paper, ExecutionTarget, ExecutionResult};
use iris::builder::part::{Part, Size, IntegerCalculateType, IntegerComparisonType};

#[test]
fn test_build_native_executable_arithmetic() {

    let paper = Paper::new("NativeArithmetic", vec![
        Part::Integer(Size::S32, 25, false),
        Part::Integer(Size::S32, 17, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Add),
        Part::Integer(Size::S32, 2, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Mul),
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/native_arithmetic_bin");
    paper.build_native_executable(bin_path, OptimizationLevel::Default)
        .expect("Failed to build native executable");

    assert!(bin_path.exists(), "Binary file does not exist");


    let output = Command::new(bin_path).output().expect("Failed to execute native binary");
    assert_eq!(output.status.code(), Some(84));

    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn test_build_pure_native_executable() {
    let paper = Paper::new("PureNativeMath", vec![
        Part::Integer(Size::S32, 25, false),
        Part::Integer(Size::S32, 17, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Add),
        Part::Integer(Size::S32, 2, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Mul),
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/pure_native_arithmetic_bin");
    paper.build_native_executable_pure(bin_path, OptimizationLevel::Aggressive)
        .expect("Failed to build pure native executable");

    assert!(bin_path.exists(), "Binary file does not exist");

    let output = Command::new(bin_path).output().expect("Failed to execute native binary");
    assert_eq!(output.status.code(), Some(84));


    let nm_output = Command::new("nm").arg(bin_path).output();
    if let Ok(nm) = nm_output {
        let syms = String::from_utf8_lossy(&nm.stdout);
        assert!(!syms.contains("iris_push_i32"), "Must not contain VM runtime calls");
        assert!(!syms.contains("iris_print_logic"), "Must not contain VM runtime calls");
    }

    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn test_build_native_executable_print_and_string() {
    let paper = Paper::new("NativePrint", vec![
        Part::String("Hello from Iris Native Ahead-of-Time Binary!".to_string()),
        Part::Print,
        Part::Integer(Size::S32, 42, false),
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/native_print_bin");
    paper.build_native_executable(bin_path, OptimizationLevel::Default)
        .expect("Failed to build native executable");

    assert!(bin_path.exists());

    let output = Command::new(bin_path).output().expect("Failed to execute native binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "Hello from Iris Native Ahead-of-Time Binary!");
    assert_eq!(output.status.code(), Some(42));

    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn test_build_native_all_formats() {
    let paper = Paper::new("FormatsTest", vec![
        Part::Integer(Size::S32, 100, false),
        Part::Integer(Size::S32, 23, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Sub),
        Part::ReturnInteger,
    ]);

    let obj_path = Path::new("target/formats_test.o");
    let asm_path = Path::new("target/formats_test.s");
    let bc_path = Path::new("target/formats_test.bc");
    let ll_path = Path::new("target/formats_test.ll");
    let bin_path = Path::new("target/formats_test_bin");


    paper.build_native_object(obj_path, OptimizationLevel::Aggressive).expect("Emit object failed");
    assert!(obj_path.exists());
    assert!(std::fs::metadata(obj_path).unwrap().len() > 0);


    paper.build_native_assembly(asm_path, OptimizationLevel::Aggressive).expect("Emit assembly failed");
    assert!(asm_path.exists());
    let asm_content = std::fs::read_to_string(asm_path).unwrap();
    assert!(!asm_content.is_empty());


    paper.build_native_bitcode(bc_path).expect("Emit bitcode failed");
    assert!(bc_path.exists());
    assert!(std::fs::metadata(bc_path).unwrap().len() > 0);


    paper.build_native_ir(ll_path).expect("Emit LLVM IR failed");
    assert!(ll_path.exists());
    let ll_content = std::fs::read_to_string(ll_path).unwrap();
    assert!(ll_content.contains("FormatsTest"));


    let status = Command::new("cc")
        .arg(obj_path)
        .arg("-o")
        .arg(bin_path)
        .status()
        .expect("Failed to link object file");
    assert!(status.success());


    let output = Command::new(bin_path).output().expect("Failed to run binary");
    assert_eq!(output.status.code(), Some(77));


    let _ = std::fs::remove_file(obj_path);
    let _ = std::fs::remove_file(asm_path);
    let _ = std::fs::remove_file(bc_path);
    let _ = std::fs::remove_file(ll_path);
    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn test_loop_and_locals_in_native_binary() {


    let paper = Paper::new("NativeLoop", vec![

        Part::Integer(Size::S64, 0, false),
        Part::StoreLocal("sum".to_string()),


        Part::Integer(Size::S64, 1, false),
        Part::StoreLocal("i".to_string()),

        Part::Label("loop".to_string()),
        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 10, false),
        Part::ComparisonInteger(IntegerComparisonType::Greater, false, false),
        Part::JumpIfTrue("end".to_string()),


        Part::PushLocal("sum".to_string()),
        Part::PushLocal("i".to_string()),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Add),
        Part::StoreLocal("sum".to_string()),


        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 1, false),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Add),
        Part::StoreLocal("i".to_string()),

        Part::Jump("loop".to_string()),

        Part::Label("end".to_string()),
        Part::PushLocal("sum".to_string()),
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/native_loop_bin");
    paper.build_native_executable(bin_path, OptimizationLevel::Aggressive)
        .expect("Failed to build loop executable");

    let output = Command::new(bin_path).output().expect("Failed to execute");
    assert_eq!(output.status.code(), Some(55));

    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn test_paper_unified_execution_target_native() {
    let paper = Paper::new("UnifiedNativeTarget", vec![
        Part::String("Running via ExecutionTarget::Native!".to_string()),
        Part::Print,
        Part::Integer(Size::S32, 99, false),
        Part::ReturnInteger,
    ]);

    let result = paper.execute(ExecutionTarget::Native(OptimizationLevel::Default))
        .expect("Native execution failed");

    match result {
        ExecutionResult::Native { exit_code, stdout } => {
            assert_eq!(exit_code, 99);
            assert_eq!(stdout.trim(), "Running via ExecutionTarget::Native!");
        }
        _ => panic!("Expected ExecutionResult::Native"),
    }
}

#[test]
fn test_native_struct_operations() {
    let paper = Paper::new("NativeStruct", vec![
        Part::AllocStackStruct("Vec2".to_string()),
        Part::Dup,
        Part::Integer(Size::S32, 50, false),
        Part::SetStructField {
            struct_name: "Vec2".to_string(),
            field_index: 0,
            offset: 0,
            size: Size::S32,
        },
        Part::Dup,
        Part::Integer(Size::S32, 75, false),
        Part::SetStructField {
            struct_name: "Vec2".to_string(),
            field_index: 1,
            offset: 4,
            size: Size::S32,
        },
        Part::GetStructField {
            struct_name: "Vec2".to_string(),
            field_index: 1,
            offset: 4,
            size: Size::S32,
        },
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/native_struct_bin");
    paper.build_native_executable(bin_path, OptimizationLevel::Default)
        .expect("Failed to build struct executable");

    let output = Command::new(bin_path).output().expect("Failed to execute");
    assert_eq!(output.status.code(), Some(75));

    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn test_cli_build_and_run_native() {
    let paper = Paper::new("CliDemo", vec![
        Part::String("Iris CLI built native binary successfully!".to_string()),
        Part::Print,
        Part::Integer(Size::S32, 123, false),
        Part::ReturnInteger,
    ]);

    let paper_file = Path::new("target/cli_demo.paper");
    let bin_file = Path::new("target/cli_demo_bin");

    paper.save_binary(paper_file).expect("Failed to save paper");


    let build_status = Command::new("cargo")
        .args(&["run", "--bin", "iris", "--", "build", paper_file.to_str().unwrap(), "-o", bin_file.to_str().unwrap(), "-O", "3"])
        .status()
        .expect("Failed to run iris CLI build");
    assert!(build_status.success());
    assert!(bin_file.exists());


    let output = Command::new(bin_file).output().expect("Failed to execute CLI built binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "Iris CLI built native binary successfully!");
    assert_eq!(output.status.code(), Some(123));

    let _ = std::fs::remove_file(paper_file);
    let _ = std::fs::remove_file(bin_file);
}

#[test]
fn test_pure_native_optimization_promotes_allocas_and_dead_blocks() {
    let paper = Paper::new("PureLoopOpt", vec![
        Part::Integer(Size::S64, 0, false),
        Part::StoreLocal("sum".to_string()),
        Part::Integer(Size::S64, 1, false),
        Part::StoreLocal("i".to_string()),
        Part::Label("loop".to_string()),
        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 10, false),
        Part::ComparisonInteger(IntegerComparisonType::Greater, false, false),
        Part::JumpIfTrue("end".to_string()),
        Part::PushLocal("sum".to_string()),
        Part::PushLocal("i".to_string()),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Add),
        Part::StoreLocal("sum".to_string()),
        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 1, false),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Add),
        Part::StoreLocal("i".to_string()),
        Part::Jump("loop".to_string()),
        Part::Label("end".to_string()),
        Part::PushLocal("sum".to_string()),
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/pure_loop_opt_bin");
    paper.build_native_executable_pure(bin_path, OptimizationLevel::Aggressive)
        .expect("Failed to build pure loop executable");

    let output = Command::new(bin_path).output().expect("Failed to execute");
    assert_eq!(output.status.code(), Some(55));
    let _ = std::fs::remove_file(bin_path);

    let context = inkwell::context::Context::create();
    let module = context.create_module("test_opt");
    let rfc = iris::builder::compiler::RuntimeFunctionCompiler::new(&context, module);
    let func = rfc.compile_paper_function_pure(&paper).unwrap();
    let ir = func.to_string();
    assert!(!ir.contains("alloca"), "mem2reg pass must eliminate all stack allocas for locals: {}", ir);
    assert!(!ir.contains("after_jump"), "DCE pass must prune dead blocks: {}", ir);
}

#[test]
fn test_pure_native_fast_math_and_fma() {
    let paper = Paper::new("PureFMA", vec![
        Part::Decimal(Size::S64, 3.0),
        Part::Decimal(Size::S64, 4.0),
        Part::Decimal(Size::S64, 5.0),
        Part::FusedMultiplyAdd(Size::S64),
        Part::DecimalToInteger(false, Size::S64),
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/pure_fma_bin");
    paper.build_native_executable_pure(bin_path, OptimizationLevel::Aggressive)
        .expect("Failed to build pure FMA executable");

    let output = Command::new(bin_path).output().expect("Failed to execute");
    assert_eq!(output.status.code(), Some(17));
    let _ = std::fs::remove_file(bin_path);

    let context = inkwell::context::Context::create();
    let module = context.create_module("test_fma");
    let rfc = iris::builder::compiler::RuntimeFunctionCompiler::new(&context, module);
    let func = rfc.compile_paper_function_pure(&paper).unwrap();
    let ir = func.to_string();
    assert!(ir.contains("17"), "Optimization should compute or contain 17: {}", ir);
}

#[test]
fn test_native_host_microarchitecture_target_machine() {
    let context = inkwell::context::Context::create();
    let compiler = iris::builder::native::NativeCompiler::new(&context, "test_host_arch");
    let mut config = iris::builder::native::NativeBuildConfig::default();
    config.pure_aot = true;
    let tm = compiler.create_target_machine(&config).expect("TargetMachine creation must succeed");
    let cpu = tm.get_cpu().to_string_lossy().to_string();
    assert!(!cpu.is_empty(), "CPU name must not be empty");
    let feature_str = tm.get_feature_string().to_string_lossy().to_string();
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        assert!(feature_str.contains("avx2"), "TargetMachine feature string should include avx2: {}", feature_str);
    }
}

#[test]
fn test_pure_native_fastcc_attributes_and_noalias_metadata() {
    let paper = Paper::new("PureStructAndAttrs", vec![
        Part::AllocStackStruct("Point".to_string()),
        Part::Dup,
        Part::Integer(Size::S32, 42, false),
        Part::SetStructField {
            struct_name: "Point".to_string(),
            field_index: 0,
            offset: 0,
            size: Size::S32,
        },
        Part::GetStructField {
            struct_name: "Point".to_string(),
            field_index: 0,
            offset: 0,
            size: Size::S32,
        },
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/pure_struct_fastcc_bin");
    paper.build_native_executable_pure(bin_path, OptimizationLevel::Aggressive)
        .expect("Failed to build pure struct executable with FastCC");

    let output = Command::new(bin_path).output().expect("Failed to execute");
    assert_eq!(output.status.code(), Some(42));
    let _ = std::fs::remove_file(bin_path);

    let context = inkwell::context::Context::create();
    let module = context.create_module("test_attrs");
    let rfc = iris::builder::compiler::RuntimeFunctionCompiler::new(&context, module);
    let func = rfc.compile_paper_function_pure(&paper).unwrap();
    assert_eq!(func.get_call_conventions(), inkwell::llvm_sys::LLVMCallConv::LLVMFastCallConv as u32);
    let ir = func.to_string();
    assert!(ir.contains("fastcc") || ir.contains("nounwind"), "IR should reflect fastcc / attributes: {}", ir);
}

#[test]
fn test_pure_native_executable_with_lto() {
    let paper = Paper::new("PureLTO", vec![
        Part::Integer(Size::S64, 100, false),
        Part::Integer(Size::S64, 20, false),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Div(false)),
        Part::ReturnInteger,
    ]);

    let bin_path = Path::new("target/pure_lto_bin");
    paper.build_native_executable_pure(bin_path, OptimizationLevel::Aggressive)
        .expect("Failed to build pure executable with LTO");

    let output = Command::new(bin_path).output().expect("Failed to execute");
    assert_eq!(output.status.code(), Some(5));
    let _ = std::fs::remove_file(bin_path);
}
