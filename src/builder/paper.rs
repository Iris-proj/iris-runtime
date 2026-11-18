use serde::{Deserialize, Serialize};
use crate::builder::part::Part;
use crate::vm::value::Value;
use crate::vm::engine::{IrisEngine, EngineError};
use crate::vm::gpu::{GpuDevice, GpuExecutionResult};
use crate::builder::compiler::RuntimeFunctionCompiler;
use inkwell::execution_engine::ExecutionEngine;


pub enum ExecutionTarget<'a, 'ctx> {
    Vm(&'a mut IrisEngine),
    Cpu(&'a RuntimeFunctionCompiler<'ctx>, &'a ExecutionEngine<'ctx>),
    CpuParallel(&'a RuntimeFunctionCompiler<'ctx>, &'a ExecutionEngine<'ctx>),
    Gpu(&'a mut GpuDevice),
    Native(inkwell::OptimizationLevel),
}


#[derive(Debug, Clone, PartialEq)]
pub struct CpuParallelExecutionResult {
    pub return_values: Vec<Option<i64>>,
    pub last_return: Option<i64>,
    pub total_threads_executed: usize,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
}


#[derive(Debug, Clone)]
pub enum ExecutionResult {
    Vm(Option<Value>),
    Cpu(Option<i64>),
    CpuParallel(CpuParallelExecutionResult),
    Gpu(GpuExecutionResult),
    Native { exit_code: i32, stdout: String },
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Paper {
    pub name: String,
    pub entry_point: String,
    pub parts: Vec<Part>,
    pub locals: Vec<Local>,
    pub stack_size: usize,
}

impl Paper {
    pub fn new(name: impl Into<String>, parts: Vec<Part>) -> Self {
        Self {
            name: name.into(),
            entry_point: "main".to_string(),
            parts,
            locals: Vec::new(),
            stack_size: 1024,
        }
    }

    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    pub fn with_stack_size(mut self, stack_size: usize) -> Self {
        self.stack_size = stack_size;
        self
    }

    
    pub fn register(&self) -> &Self {
        crate::vm::pool::register_global_paper(self.clone());
        self
    }

    pub fn registered(self) -> Self {
        self.register();
        self
    }

    pub fn add_part(&mut self, part: Part) -> &mut Self {
        self.parts.push(part);
        self
    }

    pub fn add_local(&mut self, name: impl Into<String>, ty: LocalType) -> &mut Self {
        self.locals.push(Local {
            name: name.into(),
            ty,
        });
        self
    }

    
    pub fn run_vm(&self, engine: &mut IrisEngine) -> Result<Option<Value>, EngineError> {
        engine.execute_paper(self)
    }

    
    
    pub fn run_cpu<'ctx>(
        &self,
        rfc: &RuntimeFunctionCompiler<'ctx>,
        ee: &ExecutionEngine<'ctx>,
    ) -> Result<Option<i64>, String> {
        for part in &self.parts {
            if let Part::KernelLaunch { grid_dim, block_dim, body, .. } = part {
                let sub_paper = Paper::new(format!("{}_kernel", self.name), body.clone());
                let par_res = sub_paper.run_cpu_parallel(rfc, ee, *grid_dim, *block_dim)?;
                return Ok(par_res.last_return);
            }
        }

        let compiled = rfc
            .jit_compile(self.clone(), ee)
            .ok_or_else(|| "Failed to JIT-compile Paper to native CPU machine code".to_string())?;

        let mut vm = IrisEngine::new();
        unsafe {
            compiled.call(&mut vm);
        }

        let ret = match vm.stack.pop() {
            Some(Value::I32(v)) => Some(v as i64),
            Some(Value::I64(v)) => Some(v),
            Some(Value::U32(v)) => Some(v as i64),
            Some(Value::U64(v)) => Some(v as i64),
            Some(Value::I8(v)) => Some(v as i64),
            Some(Value::I16(v)) => Some(v as i64),
            _ => None,
        };
        Ok(ret)
    }

    /// Executes this Paper in parallel across multiple CPU hardware threads using workgroups.
    pub fn run_cpu_parallel<'ctx>(
        &self,
        rfc: &RuntimeFunctionCompiler<'ctx>,
        ee: &ExecutionEngine<'ctx>,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    ) -> Result<CpuParallelExecutionResult, String> {
        let compiled = rfc
            .jit_compile(self.clone(), ee)
            .ok_or_else(|| "Failed to JIT-compile Paper to native CPU machine code".to_string())?;

        let total_blocks = (grid_dim.0 * grid_dim.1 * grid_dim.2) as usize;
        let threads_per_block = (block_dim.0 * block_dim.1 * block_dim.2) as usize;
        let total_threads = total_blocks * threads_per_block;

        if total_threads == 0 {
            return Ok(CpuParallelExecutionResult {
                return_values: Vec::new(),
                last_return: None,
                total_threads_executed: 0,
                grid_dim,
                block_dim,
            });
        }

        let func_ptr = unsafe { compiled.as_raw() };
        let func: crate::builder::compiler::Func = unsafe { std::mem::transmute(func_ptr) };

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(total_threads)
            .max(1);

        let chunk_size = (total_threads + num_workers - 1) / num_workers;

        let thread_results: Vec<Vec<(usize, Option<i64>)>> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for worker_id in 0..num_workers {
                let start = worker_id * chunk_size;
                let end = (start + chunk_size).min(total_threads);
                if start >= end {
                    continue;
                }

                handles.push(s.spawn(move || {
                    let mut local_results = Vec::with_capacity(end - start);
                    for gid in start..end {
                        let block_idx = gid / threads_per_block;
                        let thread_idx = gid % threads_per_block;

                        let mut vm = IrisEngine::new().with_thread_context(gid, thread_idx, block_idx);
                        unsafe {
                            func(&mut vm);
                        }

                        let ret = match vm.stack.pop() {
                            Some(Value::I32(v)) => Some(v as i64),
                            Some(Value::I64(v)) => Some(v),
                            Some(Value::U32(v)) => Some(v as i64),
                            Some(Value::U64(v)) => Some(v as i64),
                            Some(Value::I8(v)) => Some(v as i64),
                            Some(Value::I16(v)) => Some(v as i64),
                            _ => None,
                        };
                        local_results.push((gid, ret));
                    }
                    local_results
                }));
            }

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut return_values = vec![None; total_threads];
        for worker_res in thread_results {
            for (gid, val) in worker_res {
                return_values[gid] = val;
            }
        }

        let last_return = return_values.last().cloned().flatten();

        Ok(CpuParallelExecutionResult {
            return_values,
            last_return,
            total_threads_executed: total_threads,
            grid_dim,
            block_dim,
        })
    }

    
    pub fn run_gpu(&self, gpu: &mut GpuDevice) -> Result<GpuExecutionResult, String> {
        gpu.execute_paper(self)
    }

    
    pub fn execute<'a, 'ctx>(&self, target: ExecutionTarget<'a, 'ctx>) -> Result<ExecutionResult, String> {
        match target {
            ExecutionTarget::Vm(engine) => {
                let val = self.run_vm(engine).map_err(|e| format!("VM Execution error: {:?}", e))?;
                Ok(ExecutionResult::Vm(val))
            }
            ExecutionTarget::Cpu(rfc, ee) => {
                let val = self.run_cpu(rfc, ee)?;
                Ok(ExecutionResult::Cpu(val))
            }
            ExecutionTarget::CpuParallel(rfc, ee) => {
                let mut last_res = None;
                for part in &self.parts {
                    if let Part::KernelLaunch { grid_dim, block_dim, body, .. } = part {
                        let sub_paper = Paper::new(format!("{}_kernel", self.name), body.clone());
                        let par_res = sub_paper.run_cpu_parallel(rfc, ee, *grid_dim, *block_dim)?;
                        last_res = Some(par_res);
                    }
                }
                let res = match last_res {
                    Some(r) => r,
                    None => self.run_cpu_parallel(rfc, ee, (1, 1, 1), (1, 1, 1))?,
                };
                Ok(ExecutionResult::CpuParallel(res))
            }
            ExecutionTarget::Gpu(gpu) => {
                let val = self.run_gpu(gpu)?;
                Ok(ExecutionResult::Gpu(val))
            }
            ExecutionTarget::Native(opt) => {
                let (exit_code, stdout) = self.run_native(opt)?;
                Ok(ExecutionResult::Native { exit_code, stdout })
            }
        }
    }

    
    pub fn build_native_executable(&self, output_path: impl AsRef<std::path::Path>, opt: inkwell::OptimizationLevel) -> Result<(), String> {
        let context = inkwell::context::Context::create();
        let mut compiler = crate::builder::native::NativeCompiler::new(&context, &self.name);
        compiler.compile_paper(self)?;
        compiler.emit_native_runtime()?;
        compiler.emit_main(&self.name)?;
        let mut config = crate::builder::native::NativeBuildConfig::default();
        config.opt_level = opt;
        compiler.build_executable(output_path.as_ref(), &config)
    }

    
    pub fn build_native_object(&self, output_path: impl AsRef<std::path::Path>, opt: inkwell::OptimizationLevel) -> Result<(), String> {
        let context = inkwell::context::Context::create();
        let mut compiler = crate::builder::native::NativeCompiler::new(&context, &self.name);
        compiler.compile_paper(self)?;
        compiler.emit_native_runtime()?;
        compiler.emit_main(&self.name)?;
        let mut config = crate::builder::native::NativeBuildConfig::default();
        config.opt_level = opt;
        compiler.emit_object_file(output_path.as_ref(), &config)
    }

    
    pub fn build_native_assembly(&self, output_path: impl AsRef<std::path::Path>, opt: inkwell::OptimizationLevel) -> Result<(), String> {
        let context = inkwell::context::Context::create();
        let mut compiler = crate::builder::native::NativeCompiler::new(&context, &self.name);
        compiler.compile_paper(self)?;
        compiler.emit_native_runtime()?;
        let mut config = crate::builder::native::NativeBuildConfig::default();
        config.opt_level = opt;
        compiler.emit_assembly(output_path.as_ref(), &config)
    }

    
    pub fn build_native_bitcode(&self, output_path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let context = inkwell::context::Context::create();
        let mut compiler = crate::builder::native::NativeCompiler::new(&context, &self.name);
        compiler.compile_paper(self)?;
        compiler.emit_native_runtime()?;
        compiler.emit_bitcode(output_path.as_ref())
    }

    
    pub fn build_native_ir(&self, output_path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let context = inkwell::context::Context::create();
        let mut compiler = crate::builder::native::NativeCompiler::new(&context, &self.name);
        compiler.compile_paper(self)?;
        compiler.emit_native_runtime()?;
        compiler.emit_llvm_ir(output_path.as_ref())
    }

    
    pub fn run_native(&self, opt: inkwell::OptimizationLevel) -> Result<(i32, String), String> {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let bin_name = format!("iris_native_{}_{}", self.name, unique_id);
        let bin_path = temp_dir.join(bin_name);

        self.build_native_executable(&bin_path, opt)?;

        let output = std::process::Command::new(&bin_path)
            .output()
            .map_err(|e| {
                let _ = std::fs::remove_file(&bin_path);
                format!("Failed to execute native binary: {}", e)
            })?;

        let _ = std::fs::remove_file(&bin_path);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let code = output.status.code().unwrap_or(-1);
        Ok((code, stdout))
    }

    
    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serde::encode_to_vec(self, bincode::config::standard())?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    
    pub fn load_binary(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let (paper, _): (Self, usize) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
        Ok(paper)
    }
}

impl Default for Paper {
    fn default() -> Self {
        Self {
            name: String::new(),
            entry_point: "main".to_string(),
            parts: Vec::new(),
            locals: Vec::new(),
            stack_size: 1024,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Local {
    pub name: String,
    pub ty: LocalType,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum LocalType {
    Integer(crate::builder::part::Size),
    Decimal(crate::builder::part::Size),
    Bool,
}
