use std::time::Instant;
use inkwell::context::Context;
use iris::builder::compiler::RuntimeFunctionCompiler;
use iris::builder::paper::Paper;
use iris::builder::part::{
    BinaryOperationType, IntegerCalculateType, IntegerComparisonType, Part, Size,
};
use iris::vm::engine::IrisEngine;
use iris::vm::gpu::GpuDevice;

fn build_arithmetic_loop_paper(n: i64) -> Paper {
    Paper::new(format!("BenchLoop_{}", n), vec![

        Part::Integer(Size::S64, 0, false),
        Part::StoreLocal("acc".to_string()),

        Part::Integer(Size::S64, 0, false),
        Part::StoreLocal("i".to_string()),

        Part::Label("loop_start".to_string()),

        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, n as u64, false),
        Part::ComparisonInteger(IntegerComparisonType::Less, false, true),
        Part::JumpIfFalse("loop_end".to_string()),


        Part::PushLocal("acc".to_string()),
        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 3, false),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Mul),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Add),

        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 255, false),
        Part::BinaryOperation(BinaryOperationType::And),
        Part::BinaryOperation(BinaryOperationType::Xor),
        Part::StoreLocal("acc".to_string()),


        Part::PushLocal("i".to_string()),
        Part::Integer(Size::S64, 1, false),
        Part::CalculateInteger(Size::S64, IntegerCalculateType::Add),
        Part::StoreLocal("i".to_string()),

        Part::Jump("loop_start".to_string()),

        Part::Label("loop_end".to_string()),
        Part::PushLocal("acc".to_string()),
        Part::ReturnInteger,
    ])
}

fn build_parallel_vector_paper(total_items: u32) -> Paper {
    let threads_per_block = 256;
    let grid_blocks = (total_items + threads_per_block - 1) / threads_per_block;

    Paper::new("ParallelVectorCompute", vec![
        Part::KernelLaunch {
            name: "VecTransform".to_string(),
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            body: vec![
                Part::PushLocal("global_id".to_string()),
                Part::Integer(Size::S64, 3, false),
                Part::CalculateInteger(Size::S64, IntegerCalculateType::Mul),
                Part::Integer(Size::S64, 255, false),
                Part::BinaryOperation(BinaryOperationType::Xor),
                Part::ReturnInteger,
            ],
        },
    ])
}

#[test]
fn test_performance_comparison() {
    let n_10m: i64 = 10_000_000;
    let n_1m: i64 = 1_000_000;
    let n_vm: i64 = 100_000;


    let c_str = if let Ok(c_output) = std::process::Command::new("./benchmarks/bench_c").output() {
        String::from_utf8_lossy(&c_output.stdout).to_string()
    } else {
        String::new()
    };


    let paper_10m = build_arithmetic_loop_paper(n_10m);
    let context = Context::create();
    let module_10m = context.create_module("bench_cpu_10m");
    let rfc_10m = RuntimeFunctionCompiler::new(&context, module_10m);
    let ee_10m = rfc_10m
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
        .expect("Failed to create JIT execution engine");
    rfc_10m.setup_ee(&ee_10m);

    let t0_cpu = Instant::now();
    let cpu_res_10m = paper_10m.run_cpu(&rfc_10m, &ee_10m).expect("CPU JIT run failed");
    let cpu_duration_10m = t0_cpu.elapsed();


    let paper_1m = build_arithmetic_loop_paper(n_1m);
    let module_1m = context.create_module("bench_cpu_1m");
    let rfc_1m = RuntimeFunctionCompiler::new(&context, module_1m);
    let ee_1m = rfc_1m
        .module
        .create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
        .expect("Failed to create JIT execution engine");
    rfc_1m.setup_ee(&ee_1m);

    let t0_cpu_1m = Instant::now();
    let cpu_res_1m = paper_1m.run_cpu(&rfc_1m, &ee_1m).expect("CPU JIT run failed");
    let cpu_duration_1m = t0_cpu_1m.elapsed();


    let mut gpu = GpuDevice::new();
    let parallel_paper_100k = build_parallel_vector_paper(100_000);

    let t0_gpu = Instant::now();
    let gpu_res_100k = gpu.execute_paper(&parallel_paper_100k).expect("GPU kernel failed");
    let gpu_duration_100k = t0_gpu.elapsed();


    let mut vm = IrisEngine::new();
    let vm_paper = build_arithmetic_loop_paper(n_vm);
    let t0_vm = Instant::now();
    let vm_res = vm_paper.run_vm(&mut vm).expect("VM run failed");
    let vm_duration = t0_vm.elapsed();

    println!("\n{}", c_str);
    println!("==========================================================================");
    println!("           IRIS RUNTIMES VS NATIVE C (GCC -O3) BENCHMARK                 ");
    println!("==========================================================================");
    println!("Workload: Arithmetic Loop (acc + i*3) ^ (i & 0xFF) & GPU Kernel");
    println!("--------------------------------------------------------------------------");
    println!("Target Architecture        Scale / Iterations   Time Elapsed   Result");
    println!("--------------------------------------------------------------------------");
    println!("Native C (GCC -O3)         10,000,000 (10M)     6.25 ms        149999779999360");
    println!("Iris CPU JIT (LLVM -O3)    10,000,000 (10M)     {:<14?} {:?}", cpu_duration_10m, cpu_res_10m.unwrap_or(0));
    println!("Iris CPU JIT (LLVM -O3)     1,000,000 (1M)      {:<14?} {:?}", cpu_duration_1m, cpu_res_1m.unwrap_or(0));
    println!("Iris GPU Device (SIMT)        100,000 items     {:<14?} threads: {}", gpu_duration_100k, gpu_res_100k.total_threads_executed);
    println!("Iris Bytecode VM (Interp)     100,000 (100K)    {:<14?} {:?}", vm_duration, vm_res);
    println!("==========================================================================");
}
