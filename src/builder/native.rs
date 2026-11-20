use std::path::Path;
use std::process::Command;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::FunctionValue;
use inkwell::targets::{InitializationConfig, Target, TargetMachine, FileType, RelocMode, CodeModel};
use inkwell::OptimizationLevel;
use inkwell::AddressSpace;
use crate::builder::compiler::RuntimeFunctionCompiler;
use crate::builder::paper::Paper;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOutputFormat {

    Executable,

    Object,

    Assembly,

    Bitcode,

    LlvmIr,
}


#[derive(Debug, Clone)]
pub struct NativeBuildConfig {
    pub opt_level: OptimizationLevel,
    pub reloc_mode: RelocMode,
    pub code_model: CodeModel,
    pub target_triple: Option<String>,
    pub cpu: Option<String>,
    pub features: Option<String>,
    pub emit_main: bool,
    pub linker: Option<String>,
    pub linker_args: Vec<String>,


    pub pure_aot: bool,
}

impl Default for NativeBuildConfig {
    fn default() -> Self {
        Self {
            opt_level: OptimizationLevel::Default,
            reloc_mode: RelocMode::PIC,
            code_model: CodeModel::Default,
            target_triple: None,
            cpu: None,
            features: None,
            emit_main: true,
            linker: None,
            linker_args: Vec::new(),
            pure_aot: false,
        }
    }
}


pub struct NativeCompiler<'ctx> {
    pub context: &'ctx Context,
    pub rfc: RuntimeFunctionCompiler<'ctx>,
}

impl<'ctx> NativeCompiler<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let rfc = RuntimeFunctionCompiler::new(context, module);
        Self {
            context,
            rfc,
        }
    }

    pub fn from_rfc(rfc: RuntimeFunctionCompiler<'ctx>) -> Self {
        let context = rfc.context;
        Self {
            context,
            rfc,
        }
    }

    pub fn module(&self) -> &Module<'ctx> {
        &self.rfc.module
    }


    pub fn compile_paper(&mut self, paper: &Paper) -> Result<FunctionValue<'ctx>, String> {
        self.rfc.compile_paper_function(paper)
            .ok_or_else(|| format!("Failed to compile Paper '{}' to native LLVM IR", paper.name))
    }


    pub fn compile_paper_pure(&mut self, paper: &Paper) -> Result<FunctionValue<'ctx>, String> {
        self.rfc.compile_paper_function_pure(paper)
            .ok_or_else(|| format!("Failed to compile Paper '{}' to pure native LLVM IR", paper.name))
    }


    pub fn emit_native_runtime(&self) -> Result<(), String> {
        let i8_type = self.context.i8_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let _void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::from(0));

        let stack_array_type = i64_type.array_type(2048);
        let types_array_type = i8_type.array_type(2048);
        let engine_struct_type = self.context.struct_type(&[
            stack_array_type.into(),
            types_array_type.into(),
            i64_type.into(),
            i64_type.into(),
            i64_type.into(),
            i64_type.into(),
        ], false);


        let puts_fn = self.rfc.module.get_function("puts")
            .unwrap_or_else(|| self.rfc.module.add_function("puts", i32_type.fn_type(&[ptr_type.into()], false), None));
        let printf_fn = self.rfc.module.get_function("printf")
            .unwrap_or_else(|| self.rfc.module.add_function("printf", i32_type.fn_type(&[ptr_type.into()], true), None));


        if let Some(push_i32_fn) = self.rfc.module.get_function("iris_push_i32") {
            if push_i32_fn.count_basic_blocks() == 0 {
                let bb = self.context.append_basic_block(push_i32_fn, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);
                let eng = push_i32_fn.get_first_param().unwrap().into_pointer_value();
                let val = push_i32_fn.get_nth_param(1).unwrap().into_int_value();
                let val_i64 = builder.build_int_s_extend(val, i64_type, "ext_i64").unwrap();

                let top_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(2, false)], "top_ptr").unwrap() };
                let top = builder.build_load(i64_type, top_ptr, "top").unwrap().into_int_value();

                let stack_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(0, false), top], "stack_elem").unwrap() };
                builder.build_store(stack_elem_ptr, val_i64).unwrap();

                let type_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(1, false), top], "type_elem").unwrap() };
                builder.build_store(type_elem_ptr, i8_type.const_int(0, false)).unwrap();

                let next_top = builder.build_int_add(top, i64_type.const_int(1, false), "next_top").unwrap();
                builder.build_store(top_ptr, next_top).unwrap();
                builder.build_return(None).unwrap();
            }
        }


        if let Some(push_i64_fn) = self.rfc.module.get_function("iris_push_i64") {
            if push_i64_fn.count_basic_blocks() == 0 {
                let bb = self.context.append_basic_block(push_i64_fn, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);
                let eng = push_i64_fn.get_first_param().unwrap().into_pointer_value();
                let val = push_i64_fn.get_nth_param(1).unwrap().into_int_value();

                let top_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(2, false)], "top_ptr").unwrap() };
                let top = builder.build_load(i64_type, top_ptr, "top").unwrap().into_int_value();

                let stack_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(0, false), top], "stack_elem").unwrap() };
                builder.build_store(stack_elem_ptr, val).unwrap();

                let type_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(1, false), top], "type_elem").unwrap() };
                builder.build_store(type_elem_ptr, i8_type.const_int(0, false)).unwrap();

                let next_top = builder.build_int_add(top, i64_type.const_int(1, false), "next_top").unwrap();
                builder.build_store(top_ptr, next_top).unwrap();
                builder.build_return(None).unwrap();
            }
        }


        if let Some(push_str_fn) = self.rfc.module.get_function("iris_push_str") {
            if push_str_fn.count_basic_blocks() == 0 {
                let bb = self.context.append_basic_block(push_str_fn, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);
                let eng = push_str_fn.get_first_param().unwrap().into_pointer_value();
                let str_p = push_str_fn.get_nth_param(1).unwrap().into_pointer_value();
                let str_as_int = builder.build_ptr_to_int(str_p, i64_type, "str_int").unwrap();

                let top_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(2, false)], "top_ptr").unwrap() };
                let top = builder.build_load(i64_type, top_ptr, "top").unwrap().into_int_value();

                let stack_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(0, false), top], "stack_elem").unwrap() };
                builder.build_store(stack_elem_ptr, str_as_int).unwrap();

                let type_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(1, false), top], "type_elem").unwrap() };
                builder.build_store(type_elem_ptr, i8_type.const_int(1, false)).unwrap();

                let next_top = builder.build_int_add(top, i64_type.const_int(1, false), "next_top").unwrap();
                builder.build_store(top_ptr, next_top).unwrap();
                builder.build_return(None).unwrap();
            }
        }


        if let Some(print_fn) = self.rfc.module.get_function("iris_print_logic") {
            if print_fn.count_basic_blocks() == 0 {
                let entry_bb = self.context.append_basic_block(print_fn, "entry");
                let do_print_bb = self.context.append_basic_block(print_fn, "do_print");
                let print_str_bb = self.context.append_basic_block(print_fn, "print_str");
                let print_num_bb = self.context.append_basic_block(print_fn, "print_num");
                let end_bb = self.context.append_basic_block(print_fn, "end");

                let builder = self.context.create_builder();
                builder.position_at_end(entry_bb);
                let eng = print_fn.get_first_param().unwrap().into_pointer_value();

                let top_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(2, false)], "top_ptr").unwrap() };
                let top = builder.build_load(i64_type, top_ptr, "top").unwrap().into_int_value();
                let is_positive = builder.build_int_compare(inkwell::IntPredicate::SGT, top, i64_type.const_int(0, false), "is_pos").unwrap();
                builder.build_conditional_branch(is_positive, do_print_bb, end_bb).unwrap();

                builder.position_at_end(do_print_bb);
                let new_top = builder.build_int_sub(top, i64_type.const_int(1, false), "new_top").unwrap();
                builder.build_store(top_ptr, new_top).unwrap();

                let stack_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(0, false), new_top], "stack_elem").unwrap() };
                let val = builder.build_load(i64_type, stack_elem_ptr, "val").unwrap().into_int_value();

                let type_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(1, false), new_top], "type_elem").unwrap() };
                let ty = builder.build_load(i8_type, type_elem_ptr, "ty").unwrap().into_int_value();

                let is_str = builder.build_int_compare(inkwell::IntPredicate::EQ, ty, i8_type.const_int(1, false), "is_str").unwrap();
                builder.build_conditional_branch(is_str, print_str_bb, print_num_bb).unwrap();


                builder.position_at_end(print_str_bb);
                let s_ptr = builder.build_int_to_ptr(val, ptr_type, "s_ptr").unwrap();
                builder.build_call(puts_fn, &[s_ptr.into()], "puts_call").unwrap();
                builder.build_unconditional_branch(end_bb).unwrap();


                builder.position_at_end(print_num_bb);
                let fmt = builder.build_global_string_ptr("%ld\n", "fmt_i64").unwrap();
                builder.build_call(printf_fn, &[fmt.as_pointer_value().into(), val.into()], "printf_call").unwrap();
                builder.build_unconditional_branch(end_bb).unwrap();

                builder.position_at_end(end_bb);
                builder.build_return(Some(&i8_type.const_int(0, false))).unwrap();
            }
        }


        if let Some(fn_val) = self.rfc.module.get_function("iris_get_global_id") {
            if fn_val.count_basic_blocks() == 0 {
                let bb = self.context.append_basic_block(fn_val, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);
                let eng = fn_val.get_first_param().unwrap().into_pointer_value();
                let ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(3, false)], "gid_ptr").unwrap() };
                let val = builder.build_load(i64_type, ptr, "gid").unwrap();
                builder.build_return(Some(&val.into_int_value())).unwrap();
            }
        }


        if let Some(fn_val) = self.rfc.module.get_function("iris_get_thread_idx") {
            if fn_val.count_basic_blocks() == 0 {
                let bb = self.context.append_basic_block(fn_val, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);
                let eng = fn_val.get_first_param().unwrap().into_pointer_value();
                let ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(4, false)], "tidx_ptr").unwrap() };
                let val = builder.build_load(i64_type, ptr, "tidx").unwrap();
                builder.build_return(Some(&val.into_int_value())).unwrap();
            }
        }


        if let Some(fn_val) = self.rfc.module.get_function("iris_get_block_idx") {
            if fn_val.count_basic_blocks() == 0 {
                let bb = self.context.append_basic_block(fn_val, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);
                let eng = fn_val.get_first_param().unwrap().into_pointer_value();
                let ptr = unsafe { builder.build_gep(engine_struct_type, eng, &[i32_type.const_int(0, false), i32_type.const_int(5, false)], "bidx_ptr").unwrap() };
                let val = builder.build_load(i64_type, ptr, "bidx").unwrap();
                builder.build_return(Some(&val.into_int_value())).unwrap();
            }
        }


        if let Some(fn_val) = self.rfc.module.get_function("iris_spawn_detached") {
            if fn_val.count_basic_blocks() == 0 {
                let bb = self.context.append_basic_block(fn_val, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);
                builder.build_return(None).unwrap();
            }
        }

        Ok(())
    }


    pub fn emit_main(&self, entry_func_name: &str) -> Result<FunctionValue<'ctx>, String> {
        let i8_type = self.context.i8_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();

        let stack_array_type = i64_type.array_type(2048);
        let types_array_type = i8_type.array_type(2048);
        let engine_struct_type = self.context.struct_type(&[
            stack_array_type.into(),
            types_array_type.into(),
            i64_type.into(),
            i64_type.into(),
            i64_type.into(),
            i64_type.into(),
        ], false);

        let main_fn = if let Some(existing) = self.rfc.module.get_function("main") {
            if existing.count_basic_blocks() > 0 {
                return Ok(existing);
            }
            existing
        } else {
            let main_type = i32_type.fn_type(&[], false);
            self.rfc.module.add_function("main", main_type, None)
        };

        let entry_func = self.rfc.module.get_function(entry_func_name)
            .ok_or_else(|| format!("Entry function '{}' not found in module", entry_func_name))?;

        let bb = self.context.append_basic_block(main_fn, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(bb);

        let eng_alloca = builder.build_alloca(engine_struct_type, "native_engine").unwrap();


        let top_ptr = unsafe { builder.build_gep(engine_struct_type, eng_alloca, &[i32_type.const_int(0, false), i32_type.const_int(2, false)], "top_ptr").unwrap() };
        builder.build_store(top_ptr, i64_type.const_int(0, false)).unwrap();
        let gid_ptr = unsafe { builder.build_gep(engine_struct_type, eng_alloca, &[i32_type.const_int(0, false), i32_type.const_int(3, false)], "gid_ptr").unwrap() };
        builder.build_store(gid_ptr, i64_type.const_int(0, false)).unwrap();
        let tid_ptr = unsafe { builder.build_gep(engine_struct_type, eng_alloca, &[i32_type.const_int(0, false), i32_type.const_int(4, false)], "tid_ptr").unwrap() };
        builder.build_store(tid_ptr, i64_type.const_int(0, false)).unwrap();
        let bid_ptr = unsafe { builder.build_gep(engine_struct_type, eng_alloca, &[i32_type.const_int(0, false), i32_type.const_int(5, false)], "bid_ptr").unwrap() };
        builder.build_store(bid_ptr, i64_type.const_int(0, false)).unwrap();


        builder.build_call(entry_func, &[eng_alloca.into()], "call_entry").unwrap();


        let top = builder.build_load(i64_type, top_ptr, "top").unwrap().into_int_value();
        let has_return = builder.build_int_compare(inkwell::IntPredicate::SGT, top, i64_type.const_int(0, false), "has_ret").unwrap();

        let ret_val_bb = self.context.append_basic_block(main_fn, "ret_val");
        let ret_zero_bb = self.context.append_basic_block(main_fn, "ret_zero");
        builder.build_conditional_branch(has_return, ret_val_bb, ret_zero_bb).unwrap();


        builder.position_at_end(ret_val_bb);
        let last_idx = builder.build_int_sub(top, i64_type.const_int(1, false), "last_idx").unwrap();
        let stack_elem_ptr = unsafe { builder.build_gep(engine_struct_type, eng_alloca, &[i32_type.const_int(0, false), i32_type.const_int(0, false), last_idx], "last_elem").unwrap() };
        let ret_val = builder.build_load(i64_type, stack_elem_ptr, "ret_val").unwrap().into_int_value();
        let ret_i32 = builder.build_int_truncate(ret_val, i32_type, "ret_i32").unwrap();
        builder.build_return(Some(&ret_i32)).unwrap();


        builder.position_at_end(ret_zero_bb);
        builder.build_return(Some(&i32_type.const_int(0, false))).unwrap();

        Ok(main_fn)
    }


    pub fn emit_main_pure(&self, entry_func_name: &str) -> Result<FunctionValue<'ctx>, String> {
        let i32_type = self.context.i32_type();

        let main_fn = if let Some(existing) = self.rfc.module.get_function("main") {
            if existing.count_basic_blocks() > 0 {
                return Ok(existing);
            }
            existing
        } else {
            let main_type = i32_type.fn_type(&[], false);
            self.rfc.module.add_function("main", main_type, None)
        };

        let entry_func = self.rfc.module.get_function(entry_func_name)
            .ok_or_else(|| format!("Entry function '{}' not found in module", entry_func_name))?;

        let bb = self.context.append_basic_block(main_fn, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(bb);

        let call_res = builder.build_call(entry_func, &[], "call_entry").unwrap();
        call_res.set_call_convention(inkwell::llvm_sys::LLVMCallConv::LLVMFastCallConv as u32);
        let ret_val = match call_res.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(val) => {
                let iv = val.into_int_value();
                if iv.get_type() == i32_type {
                    iv
                } else {
                    builder.build_int_truncate(iv, i32_type, "ret_trunc").unwrap()
                }
            }
            _ => i32_type.const_int(0, false),
        };

        builder.build_return(Some(&ret_val)).unwrap();
        Ok(main_fn)
    }


    pub fn create_target_machine(&self, config: &NativeBuildConfig) -> Result<TargetMachine, String> {
        Target::initialize_all(&InitializationConfig::default());
        let triple = if let Some(ref t) = config.target_triple {
            inkwell::targets::TargetTriple::create(t)
        } else {
            TargetMachine::get_default_triple()
        };

        let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;

        let cpu = if let Some(ref c) = config.cpu {
            c.clone()
        } else {
            let host_cpu = TargetMachine::get_host_cpu_name().to_string_lossy().to_string();
            if host_cpu.is_empty() || host_cpu == "generic" {
                "native".to_string()
            } else {
                host_cpu
            }
        };

        let mut features = if let Some(ref f) = config.features {
            f.clone()
        } else {
            TargetMachine::get_host_cpu_features().to_string_lossy().to_string()
        };

        #[cfg(target_arch = "x86_64")]
        if config.features.is_none() {
            let mut extra_features = Vec::new();
            if is_x86_feature_detected!("avx2") && !features.contains("avx2") {
                extra_features.push("+avx2");
            }
            if is_x86_feature_detected!("fma") && !features.contains("fma") {
                extra_features.push("+fma");
            }
            if is_x86_feature_detected!("bmi1") && !features.contains("bmi") {
                extra_features.push("+bmi");
            }
            if is_x86_feature_detected!("bmi2") && !features.contains("bmi2") {
                extra_features.push("+bmi2");
            }
            if is_x86_feature_detected!("avx512f") && !features.contains("avx512f") {
                extra_features.push("+avx512f");
            }
            if !extra_features.is_empty() {
                if !features.is_empty() && !features.ends_with(',') {
                    features.push(',');
                }
                features.push_str(&extra_features.join(","));
            }
        }

        target.create_target_machine(
            &triple,
            &cpu,
            &features,
            config.opt_level,
            config.reloc_mode,
            config.code_model,
        ).ok_or_else(|| "Failed to create LLVM TargetMachine".to_string())
    }


    pub fn emit_object_file(&self, output_path: &Path, config: &NativeBuildConfig) -> Result<(), String> {
        let tm = self.create_target_machine(config)?;
        if config.pure_aot {
            let options = inkwell::passes::PassBuilderOptions::create();
            options.set_loop_vectorization(true);
            options.set_loop_slp_vectorization(true);
            options.set_loop_unrolling(true);
            options.set_merge_functions(true);
            options.set_verify_each(false);
            let _ = self.rfc.module.run_passes("default<O3>", &tm, options);
        }
        tm.write_to_file(&self.rfc.module, FileType::Object, output_path)
            .map_err(|e| format!("Failed to write object file: {:?}", e))
    }


    pub fn emit_assembly(&self, output_path: &Path, config: &NativeBuildConfig) -> Result<(), String> {
        let tm = self.create_target_machine(config)?;
        if config.pure_aot {
            let options = inkwell::passes::PassBuilderOptions::create();
            options.set_loop_vectorization(true);
            options.set_loop_slp_vectorization(true);
            options.set_loop_unrolling(true);
            options.set_merge_functions(true);
            options.set_verify_each(false);
            let _ = self.rfc.module.run_passes("default<O3>", &tm, options);
        }
        tm.write_to_file(&self.rfc.module, FileType::Assembly, output_path)
            .map_err(|e| format!("Failed to write assembly file: {:?}", e))
    }


    pub fn emit_bitcode(&self, output_path: &Path) -> Result<(), String> {
        if self.rfc.module.write_bitcode_to_path(output_path) {
            Ok(())
        } else {
            Err("Failed to write bitcode to path".to_string())
        }
    }


    pub fn emit_llvm_ir(&self, output_path: &Path) -> Result<(), String> {
        self.rfc.module.print_to_file(output_path)
            .map_err(|e| format!("Failed to write LLVM IR: {:?}", e))
    }


    pub fn build_executable(&self, output_path: &Path, config: &NativeBuildConfig) -> Result<(), String> {
        let obj_path = output_path.with_extension("tmp.o");
        self.emit_object_file(&obj_path, config)?;

        let linker_cmd = config.linker.as_deref().unwrap_or("cc");
        let mut cmd = Command::new(linker_cmd);
        cmd.arg(&obj_path);
        cmd.arg("-o");
        cmd.arg(output_path);

        if config.pure_aot || config.opt_level == OptimizationLevel::Aggressive {
            cmd.arg("-O3");
            cmd.arg("-flto");
            cmd.arg("-march=native");
        }

        for arg in &config.linker_args {
            cmd.arg(arg);
        }

        let status = cmd.status().map_err(|e| {
            let _ = std::fs::remove_file(&obj_path);
            format!("Failed to invoke linker '{}': {}", linker_cmd, e)
        })?;

        let _ = std::fs::remove_file(&obj_path);

        if !status.success() {
            return Err(format!("Linker '{}' failed with status: {:?}", linker_cmd, status.code()));
        }

        Ok(())
    }


    pub fn build(&self, output_path: &Path, format: NativeOutputFormat, config: &NativeBuildConfig) -> Result<(), String> {
        match format {
            NativeOutputFormat::Executable => self.build_executable(output_path, config),
            NativeOutputFormat::Object => self.emit_object_file(output_path, config),
            NativeOutputFormat::Assembly => self.emit_assembly(output_path, config),
            NativeOutputFormat::Bitcode => self.emit_bitcode(output_path),
            NativeOutputFormat::LlvmIr => self.emit_llvm_ir(output_path),
        }
    }
}


