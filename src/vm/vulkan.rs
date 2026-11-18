use ash::vk;
use std::ffi::CStr;
use std::process::Command;
use std::io::Write;
use crate::builder::paper::Paper;
use crate::builder::part::{Part, IntegerCalculateType};
use crate::vm::value::Value;
use crate::vm::gpu::GpuExecutionResult;

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub device_name: String,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VulkanContext {
    pub fn new() -> Option<Self> {
        let entry = unsafe { ash::Entry::load().ok()? };

        let app_name = b"iris_vm\0";
        let engine_name = b"iris\0";
        let app_info = vk::ApplicationInfo::default()
            .application_name(unsafe { CStr::from_bytes_with_nul_unchecked(app_name) })
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(unsafe { CStr::from_bytes_with_nul_unchecked(engine_name) })
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_create_info, None).ok()? };

        let pdevices = unsafe { instance.enumerate_physical_devices().ok()? };
        if pdevices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return None;
        }

        
        let mut chosen = None;
        for &pdev in &pdevices {
            let props = unsafe { instance.get_physical_device_properties(pdev) };
            let queue_family_props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
            let mut compute_family = None;
            for (idx, qfp) in queue_family_props.iter().enumerate() {
                if qfp.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    compute_family = Some(idx as u32);
                    break;
                }
            }

            if let Some(q_family) = compute_family {
                let name = unsafe {
                    CStr::from_ptr(props.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                if is_discrete || chosen.is_none() {
                    chosen = Some((pdev, q_family, name));
                    if is_discrete {
                        break;
                    }
                }
            }
        }

        let (physical_device, queue_family_index, device_name) = match chosen {
            Some(c) => c,
            None => {
                unsafe { instance.destroy_instance(None) };
                return None;
            }
        };

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let queue_infos = [queue_info];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);

        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(d) => d,
            Err(_) => {
                unsafe { instance.destroy_instance(None) };
                return None;
            }
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Some(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            device_name,
            memory_properties,
        })
    }

    pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Option<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & properties) == properties
            {
                return Some(i);
            }
        }
        None
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| format!("Vulkan create_buffer error: {:?}", e))?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = self
            .find_memory_type(mem_reqs.memory_type_bits, properties)
            .ok_or_else(|| "Failed to find suitable Vulkan memory type".to_string())?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| format!("Vulkan allocate_memory error: {:?}", e))?
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| format!("Vulkan bind_buffer_memory error: {:?}", e))?;
        }

        Ok((buffer, memory))
    }

    pub fn execute_paper(
        &mut self,
        paper: &Paper,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    ) -> Result<GpuExecutionResult, String> {
        let glsl_code = generate_glsl(paper, block_dim);
        let spirv = compile_glsl_to_spirv(&glsl_code)?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .map_err(|e| format!("Vulkan create_shader_module error: {:?}", e))?
        };

        
        
        
        
        
        let buffer_size = 4096 as vk::DeviceSize;
        let (out_buffer, out_memory) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        
        unsafe {
            let data_ptr = self
                .device
                .map_memory(out_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("Vulkan map_memory error: {:?}", e))? as *mut u8;
            std::ptr::write_bytes(data_ptr, 0, buffer_size as usize);
            self.device.unmap_memory(out_memory);
        }

        
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let bindings = [binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| format!("create_descriptor_set_layout error: {:?}", e))?
        };

        
        let set_layouts = [desc_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| format!("create_pipeline_layout error: {:?}", e))?
        };

        
        let main_fn_name = b"main\0";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(unsafe { CStr::from_bytes_with_nul_unchecked(main_fn_name) });

        let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info], None)
                .map_err(|(_, e)| format!("create_compute_pipelines error: {:?}", e))?[0]
        };

        
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let desc_pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("create_descriptor_pool error: {:?}", e))?
        };

        let desc_layouts = [desc_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&desc_layouts);
        let desc_set = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("allocate_descriptor_sets error: {:?}", e))?[0]
        };

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(out_buffer)
            .offset(0)
            .range(buffer_size);

        let buffer_infos = [buffer_info];
        let write_desc = vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos);

        unsafe {
            self.device.update_descriptor_sets(&[write_desc], &[]);
        }

        
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe {
            self.device
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| format!("create_command_pool error: {:?}", e))?
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe {
            self.device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| format!("allocate_command_buffers error: {:?}", e))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(cmd_buf, &begin_info).unwrap();
            self.device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            self.device.cmd_dispatch(cmd_buf, grid_dim.0.max(1), grid_dim.1.max(1), grid_dim.2.max(1));
            self.device.end_command_buffer(cmd_buf).unwrap();
        }

        
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None).unwrap() };

        let cmd_bufs = [cmd_buf];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        unsafe {
            self.device.queue_submit(self.queue, &[submit_info], fence).unwrap();
            self.device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        }

        
        let total_threads = (grid_dim.0.max(1) * grid_dim.1.max(1) * grid_dim.2.max(1) * block_dim.0.max(1) * block_dim.1.max(1) * block_dim.2.max(1)) as usize;
        let mut ret_val = None;
        unsafe {
            let data_ptr = self
                .device
                .map_memory(out_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .unwrap() as *const u8;

            let last_thread = total_threads.saturating_sub(1).min(255);
            let return_val = std::ptr::read_unaligned((data_ptr as *const i32).add(last_thread));
            let has_return = std::ptr::read_unaligned(data_ptr.add(1024) as *const u32);
            let print_len = std::ptr::read_unaligned(data_ptr.add(1028) as *const u32);

            if print_len > 0 {
                let chars_slice = std::slice::from_raw_parts(data_ptr.add(1032), print_len as usize);
                if let Ok(s) = std::str::from_utf8(chars_slice) {
                    println!("{}", s);
                }
            }

            if has_return != 0 {
                ret_val = Some(Value::I32(return_val));
            }

            self.device.unmap_memory(out_memory);
        }

        
        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.destroy_command_pool(cmd_pool, None);
            self.device.destroy_descriptor_pool(desc_pool, None);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(desc_layout, None);
            self.device.destroy_shader_module(shader_module, None);
            self.device.destroy_buffer(out_buffer, None);
            self.device.free_memory(out_memory, None);
        }

        let total_threads = (grid_dim.0.max(1) * grid_dim.1.max(1) * grid_dim.2.max(1) * block_dim.0.max(1) * block_dim.1.max(1) * block_dim.2.max(1)) as usize;
        Ok(GpuExecutionResult {
            return_value: ret_val,
            total_threads_executed: total_threads,
            grid_dim,
            block_dim,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn compile_glsl_to_spirv(glsl: &str) -> Result<Vec<u32>, String> {
    let mut child = Command::new("glslc")
        .args(["-fshader-stage=compute", "-o", "-", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to invoke glslc: {}", e))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(glsl.as_bytes())
            .map_err(|e| format!("Failed to write to glslc: {}", e))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for glslc: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "glslc compilation failed:\n{}\nSource:\n{}",
            String::from_utf8_lossy(&output.stderr),
            glsl
        ));
    }

    let bytes = output.stdout;
    if bytes.len() % 4 != 0 {
        return Err("glslc output is not aligned to 4 bytes".to_string());
    }

    let mut spirv = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        spirv.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }

    Ok(spirv)
}

fn generate_glsl(paper: &Paper, block_dim: (u32, u32, u32)) -> String {
    let mut s = String::new();
    s.push_str("#version 450\n");
    s.push_str(&format!(
        "layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n\n",
        block_dim.0.max(1),
        block_dim.1.max(1),
        block_dim.2.max(1)
    ));

    s.push_str("layout(std430, binding = 0) buffer ResultBuffer {\n");
    s.push_str("    int return_vals[256];\n");
    s.push_str("    uint has_return;\n");
    s.push_str("    uint print_len;\n");
    s.push_str("    uint print_chars[256];\n");
    s.push_str("};\n\n");

    s.push_str("void main() {\n");
    s.push_str("    uint global_id = gl_GlobalInvocationID.x;\n");
    s.push_str("    uint thread_idx = gl_LocalInvocationID.x;\n");
    s.push_str("    uint block_idx = gl_WorkGroupID.x;\n\n");

    s.push_str("    int stack[64];\n");
    s.push_str("    int sp = 0;\n");
    s.push_str("    int struct_mem[32];\n\n");

    
    let parts_to_emit = if let Some(Part::KernelLaunch { body, .. }) = paper.parts.iter().find(|p| matches!(p, Part::KernelLaunch { .. })) {
        body.as_slice()
    } else {
        paper.parts.as_slice()
    };

    let mut pending_string: Option<String> = None;

    for part in parts_to_emit {
        match part {
            Part::String(str_val) => {
                pending_string = Some(str_val.clone());
            }
            Part::Print => {
                if let Some(ref str_val) = pending_string {
                    
                    s.push_str("    if (global_id == 0) {\n");
                    s.push_str(&format!("        print_len = {}u;\n", str_val.len()));
                    for (i, byte) in str_val.bytes().enumerate() {
                        let word_idx = i / 4;
                        let byte_offset = (i % 4) * 8;
                        s.push_str(&format!("        print_chars[{}] |= ({}u << {});\n", word_idx, byte as u32, byte_offset));
                    }
                    s.push_str("    }\n");
                } else {
                    s.push_str("    if (global_id == 0 && sp > 0) {\n");
                    s.push_str("        int val = stack[--sp];\n");
                    s.push_str("        return_val = val;\n");
                    s.push_str("        has_return = 1u;\n");
                    s.push_str("    }\n");
                }
            }
            Part::Integer(_size, val, _) => {
                s.push_str(&format!("    stack[sp++] = {};\n", *val as i32));
            }
            Part::PushLocal(name) => {
                if name == "global_id" {
                    s.push_str("    stack[sp++] = int(global_id);\n");
                } else if name == "thread_idx" {
                    s.push_str("    stack[sp++] = int(thread_idx);\n");
                } else if name == "block_idx" {
                    s.push_str("    stack[sp++] = int(block_idx);\n");
                } else {
                    s.push_str("    stack[sp++] = 0;\n");
                }
            }
            Part::CalculateInteger(_size, op) => {
                s.push_str("    {\n");
                s.push_str("        int b = stack[--sp];\n");
                s.push_str("        int a = stack[--sp];\n");
                match op {
                    IntegerCalculateType::Add => s.push_str("        stack[sp++] = a + b;\n"),
                    IntegerCalculateType::Sub => s.push_str("        stack[sp++] = a - b;\n"),
                    IntegerCalculateType::Mul => s.push_str("        stack[sp++] = a * b;\n"),
                    IntegerCalculateType::Div(_) => s.push_str("        stack[sp++] = (b != 0) ? (a / b) : 0;\n"),
                    IntegerCalculateType::Rem(_) => s.push_str("        stack[sp++] = (b != 0) ? (a % b) : 0;\n"),
                }
                s.push_str("    }\n");
            }
            Part::ReturnInteger => {
                s.push_str("    if (sp > 0) {\n");
                s.push_str("        int res = stack[--sp];\n");
                s.push_str("        return_vals[global_id % 256] = res;\n");
                s.push_str("        has_return = 1u;\n");
                s.push_str("    }\n");
                s.push_str("    return;\n");
            }
            Part::ReturnVoid => {
                s.push_str("    return;\n");
            }
            Part::AllocStackStruct(_) | Part::AllocStruct(_) => {
                s.push_str("    stack[sp++] = 0;\n"); 
            }
            Part::Dup => {
                s.push_str("    if (sp > 0) { int top = stack[sp - 1]; stack[sp++] = top; }\n");
            }
            Part::SetStructField { offset, .. } => {
                let word_offset = offset / 4;
                s.push_str("    {\n");
                s.push_str("        int val = stack[--sp];\n");
                s.push_str("        int _ptr = stack[--sp];\n");
                s.push_str(&format!("        struct_mem[{}] = val;\n", word_offset));
                s.push_str("    }\n");
            }
            Part::GetStructField { offset, .. } => {
                let word_offset = offset / 4;
                s.push_str("    {\n");
                s.push_str("        int _ptr = stack[--sp];\n");
                s.push_str(&format!("        stack[sp++] = struct_mem[{}];\n", word_offset));
                s.push_str("    }\n");
            }
            _ => {}
        }
    }

    s.push_str("}\n");
    s
}
