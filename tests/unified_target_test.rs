use inkwell::context::Context;
use iris::builder::compiler::RuntimeFunctionCompiler;
use iris::builder::paper::{Paper, ExecutionTarget, ExecutionResult};
use iris::builder::part::{
    IntegerCalculateType, IntegerComparisonType, Part, Size,
};
use iris::vm::engine::IrisEngine;
use iris::vm::gpu::GpuDevice;
use iris::vm::value::Value;

#[test]
fn test_same_paper_on_vm_cpu_and_gpu() {
    
    let paper = Paper::new("ComputeArithmetic", vec![
        Part::Integer(Size::S32, 25, false),
        Part::Integer(Size::S32, 17, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Add),
        Part::Integer(Size::S32, 2, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Mul),
        Part::ReturnInteger,
    ]);

    
    let mut vm = IrisEngine::new();
    let vm_res = paper.run_vm(&mut vm).expect("VM execution failed");
    assert_eq!(vm_res, Some(Value::I32(84)));

    
    let context = Context::create();
    let module = context.create_module("compute_cpu");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .expect("Failed to create JIT engine");
    rfc.setup_ee(&ee);

    let cpu_res = paper.run_cpu(&rfc, &ee).expect("CPU execution failed");
    assert_eq!(cpu_res, Some(84));

    
    let mut gpu = GpuDevice::new();
    let gpu_res = paper.run_gpu(&mut gpu).expect("GPU execution failed");
    assert_eq!(gpu_res.return_value, Some(Value::I32(84)));

    
    let exec_vm = paper.execute(ExecutionTarget::Vm(&mut vm)).unwrap();
    let exec_cpu = paper.execute(ExecutionTarget::Cpu(&rfc, &ee)).unwrap();
    let exec_gpu = paper.execute(ExecutionTarget::Gpu(&mut gpu)).unwrap();

    match exec_vm {
        ExecutionResult::Vm(val) => assert_eq!(val, Some(Value::I32(84))),
        _ => panic!("Expected VM result"),
    }
    match exec_cpu {
        ExecutionResult::Cpu(val) => assert_eq!(val, Some(84)),
        _ => panic!("Expected CPU result"),
    }
    match exec_gpu {
        ExecutionResult::Gpu(res) => assert_eq!(res.return_value, Some(Value::I32(84))),
        _ => panic!("Expected GPU result"),
    }
}

#[test]
fn test_struct_operations_across_targets() {
    
    let paper = Paper::new("StructTest", vec![
        Part::AllocStackStruct("Point".to_string()),
        Part::Dup,
        Part::Integer(Size::S32, 100, false),
        Part::SetStructField {
            struct_name: "Point".to_string(),
            field_index: 0,
            offset: 0,
            size: Size::S32,
        },
        Part::Dup,
        Part::Integer(Size::S32, 250, false),
        Part::SetStructField {
            struct_name: "Point".to_string(),
            field_index: 1,
            offset: 4,
            size: Size::S32,
        },
        Part::GetStructField {
            struct_name: "Point".to_string(),
            field_index: 1,
            offset: 4,
            size: Size::S32,
        },
        Part::ReturnInteger,
    ]);

    
    let mut vm = IrisEngine::new();
    let vm_res = paper.run_vm(&mut vm).expect("VM struct execution failed");
    assert_eq!(vm_res, Some(Value::I32(250)));

    
    let context = Context::create();
    let module = context.create_module("struct_cpu");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .expect("Failed to create JIT engine");
    rfc.setup_ee(&ee);

    let cpu_res = paper.run_cpu(&rfc, &ee).expect("CPU struct execution failed");
    assert_eq!(cpu_res, Some(250));

    
    let mut gpu = GpuDevice::new();
    let gpu_res = paper.run_gpu(&mut gpu).expect("GPU struct execution failed");
    assert_eq!(gpu_res.return_value, Some(Value::I32(250)));
}

#[test]
fn test_gpu_kernel_launch_and_device_memory() {
    let mut gpu = GpuDevice::new();
    gpu.device_alloc("output_tensor", 1024);

    let host_data = vec![1u8, 2, 3, 4, 5];
    gpu.copy_to_device("output_tensor", &host_data).unwrap();

    let kernel_paper = Paper::new("ParallelKernel", vec![
        Part::KernelLaunch {
            name: "VectorDouble".to_string(),
            grid_dim: (2, 1, 1),
            block_dim: (4, 1, 1),
            body: vec![
                Part::PushLocal("global_id".to_string()),
                Part::Integer(Size::S32, 2, false),
                Part::CalculateInteger(Size::S32, IntegerCalculateType::Mul),
                Part::ReturnInteger,
            ],
        },
    ]);

    let res = gpu.execute_paper(&kernel_paper).unwrap();
    
    assert_eq!(res.total_threads_executed, 8);

    let read_back = gpu.copy_from_device("output_tensor").unwrap();
    assert_eq!(&read_back[..5], &[1, 2, 3, 4, 5]);
}

#[test]
fn test_same_kernel_paper_on_vm_cpu_and_gpu() {
    
    
    
    let kernel_paper = Paper::new("UnifiedKernel", vec![
        Part::KernelLaunch {
            name: "DoubleThreadId".to_string(),
            grid_dim: (2, 1, 1),
            block_dim: (4, 1, 1),
            body: vec![
                Part::PushLocal("global_id".to_string()),
                Part::Integer(Size::S32, 2, false),
                Part::CalculateInteger(Size::S32, IntegerCalculateType::Mul),
                Part::ReturnInteger,
            ],
        },
    ]);

    
    let mut gpu = GpuDevice::new();
    let gpu_res = kernel_paper.run_gpu(&mut gpu).expect("GPU kernel failed");
    assert_eq!(gpu_res.total_threads_executed, 8);
    assert_eq!(gpu_res.return_value, Some(Value::I32(14)));

    
    let mut vm = IrisEngine::new();
    let vm_res = kernel_paper.run_vm(&mut vm).expect("VM kernel failed");
    assert_eq!(vm_res, Some(Value::I32(14)));

    
    let context = Context::create();
    let module = context.create_module("kernel_cpu");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .expect("Failed to create JIT engine");
    rfc.setup_ee(&ee);

    let cpu_res = kernel_paper.run_cpu(&rfc, &ee).expect("CPU kernel failed");
    assert_eq!(cpu_res, Some(14));

    
    let parallel_exec = kernel_paper.execute(ExecutionTarget::CpuParallel(&rfc, &ee)).unwrap();
    match parallel_exec {
        ExecutionResult::CpuParallel(par) => {
            assert_eq!(par.total_threads_executed, 8);
            assert_eq!(par.last_return, Some(14));
            for (i, val) in par.return_values.iter().enumerate() {
                assert_eq!(*val, Some((i * 2) as i64));
            }
        }
        _ => panic!("Expected CpuParallel result"),
    }
}

#[test]
fn test_cpu_multi_threaded_parallel_workgroups() {
    
    let kernel = Paper::new("CpuParTest", vec![
        Part::PushLocal("global_id".to_string()),
        Part::Integer(Size::S32, 3, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Mul),
        Part::Integer(Size::S32, 10, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Add),
        Part::ReturnInteger,
    ]);

    let context = Context::create();
    let module = context.create_module("par_workgroups");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .expect("Failed to create JIT engine");
    rfc.setup_ee(&ee);

    
    let result = kernel.run_cpu_parallel(&rfc, &ee, (4, 1, 1), (16, 1, 1)).expect("CPU parallel failed");

    assert_eq!(result.total_threads_executed, 64);
    assert_eq!(result.return_values.len(), 64);

    for i in 0..64 {
        let expected = (i * 3 + 10) as i64;
        assert_eq!(result.return_values[i], Some(expected), "Mismatch at thread {}", i);
    }
    assert_eq!(result.last_return, Some((63 * 3 + 10) as i64));
}

#[test]
fn test_fire_and_forget_spawn_detached() {
    
    let _fn2 = Paper::new("BackgroundWorker", vec![
        Part::Integer(Size::S32, 100, false),
        Part::ReturnVoid,
    ]).registered();

    
    let fn1 = Paper::new("MainController", vec![
        Part::SpawnDetached("BackgroundWorker".to_string()),
        Part::SpawnDetached("BackgroundWorker".to_string()),
        Part::Integer(Size::S32, 42, false),
        Part::ReturnInteger,
    ]);

    
    let mut vm = IrisEngine::new();
    let vm_res = fn1.run_vm(&mut vm).expect("VM execution failed");
    
    assert_eq!(vm_res, Some(Value::I32(42)));

    
    let context = Context::create();
    let module = context.create_module("spawn_detached_cpu");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .expect("Failed to create JIT engine");
    rfc.setup_ee(&ee);

    let cpu_res = fn1.run_cpu(&rfc, &ee).expect("CPU JIT execution failed");
    assert_eq!(cpu_res, Some(42));
}

#[test]
fn test_paper_print_hi() {
    let paper = Paper::new("PrintHi", vec![
        Part::String("hi".to_string()),
        Part::Print,
        Part::ReturnVoid,
    ]);

    
    let mut vm = IrisEngine::new();
    let vm_res = paper.run_vm(&mut vm);
    assert!(vm_res.is_ok());

    
    let context = Context::create();
    let module = context.create_module("print_hi_cpu");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .expect("Failed to create JIT engine");
    rfc.setup_ee(&ee);
    let cpu_res = paper.run_cpu(&rfc, &ee);
    assert!(cpu_res.is_ok());

    
    let mut gpu = GpuDevice::new();
    println!("GPU target detected hardware: {:?} (name: {:?})", gpu.is_hardware(), gpu.hardware_name());
    let gpu_res = paper.run_gpu(&mut gpu);
    assert!(gpu_res.is_ok());
}

#[test]
fn test_vulkan_physical_gpu_execution() {
    let mut gpu = GpuDevice::new();
    assert!(gpu.is_hardware(), "Physical GPU (Vulkan) not detected");
    println!("Running on Physical GPU: {}", gpu.hardware_name().unwrap_or("Unknown"));

    let compute_paper = Paper::new("GpuCompute", vec![
        Part::Integer(Size::S32, 50, false),
        Part::Integer(Size::S32, 34, false),
        Part::CalculateInteger(Size::S32, IntegerCalculateType::Add),
        Part::ReturnInteger,
    ]);

    let res = gpu.execute_paper(&compute_paper).expect("Vulkan execution failed");
    assert_eq!(res.return_value, Some(Value::I32(84)));
}

#[test]
fn test_gpu_200_cores_print_id() {
    let mut gpu = GpuDevice::new();
    gpu.vulkan = None;

    let kernel = Paper::new("PrintCoreId", vec![
        Part::PushLocal("global_id".to_string()),
        Part::Print,
        Part::ReturnVoid,
    ]);

    let paper = Paper::new("Launch200kCoresGpu", vec![
        Part::KernelLaunch {
            name: "PrintCoreIdKernel".to_string(),
            grid_dim: (2000, 1, 1),
            block_dim: (100, 1, 1),
            body: kernel.parts,
        },
    ]);

    let res = gpu.execute_paper(&paper).expect("GPU kernel launch failed");
    assert_eq!(res.total_threads_executed, 200_000);
}

#[test]
fn test_generate_hello_100k_paper() {
    let paper = Paper::new("Hello100k", vec![
        Part::Integer(Size::S64, 0, false),
        Part::StoreLocal("i".to_string()),

        Part::Label("loop".to_string()),
        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 100000, false),
        Part::ComparisonInteger(IntegerComparisonType::GreaterEqual, false, false),
        Part::JumpIfTrue("end".to_string()),

        Part::String("Hello, World!".to_string()),
        Part::Print,

        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 1, false),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Add),
        Part::StoreLocal("i".to_string()),

        Part::Jump("loop".to_string()),

        Part::Label("end".to_string()),
        Part::ReturnVoid,
    ]);

    paper.save_binary(std::path::Path::new("hello_100k.paper")).expect("Failed to write hello_100k.paper");
}

#[test]
fn test_cpu_200_cores_print_id() {
    let kernel = Paper::new("PrintCoreIdCpu", vec![
        Part::PushLocal("global_id".to_string()),
        Part::Print,
        Part::ReturnVoid,
    ]);

    let paper = Paper::new("Launch200kCoresCpu", vec![
        Part::KernelLaunch {
            name: "PrintCoreIdKernelCpu".to_string(),
            grid_dim: (2000, 1, 1),
            block_dim: (100, 1, 1),
            body: kernel.parts,
        },
    ]);

    let context = Context::create();
    let module = context.create_module("cpu_cores_module");
    let rfc = RuntimeFunctionCompiler::new(&context, module);
    let ee = rfc
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .expect("Failed to create JIT engine");
    rfc.setup_ee(&ee);

    let res = paper.execute(ExecutionTarget::CpuParallel(&rfc, &ee)).expect("CPU parallel launch failed");
    match res {
        ExecutionResult::CpuParallel(par) => {
            assert_eq!(par.total_threads_executed, 200_000);
        }
        _ => panic!("Expected CpuParallel result"),
    }
}
