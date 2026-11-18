use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::builder::paper::Paper;
use crate::builder::part::{
    BinaryOperationType, DecimalCalculateType, DecimalComparisonType, IntegerCalculateType,
    IntegerComparisonType, Part, Size, UnaryMathType,
};
use crate::vm::value::Value;

#[derive(Debug, Clone)]
pub struct GpuExecutionResult {
    pub return_value: Option<Value>,
    pub total_threads_executed: usize,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
}



pub struct GpuDevice {
    pub device_memory: HashMap<String, Vec<u8>>,
    pub max_threads_per_block: u32,
    pub vulkan: Option<crate::vm::vulkan::VulkanContext>,
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device_memory_count", &self.device_memory.len())
            .field("max_threads_per_block", &self.max_threads_per_block)
            .field("hardware_name", &self.hardware_name())
            .finish()
    }
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice {
    pub fn new() -> Self {
        let vulkan = crate::vm::vulkan::VulkanContext::new();
        Self {
            device_memory: HashMap::new(),
            max_threads_per_block: 1024,
            vulkan,
        }
    }

    pub fn new_software() -> Self {
        Self {
            device_memory: HashMap::new(),
            max_threads_per_block: 1024,
            vulkan: None,
        }
    }

    pub fn is_hardware(&self) -> bool {
        self.vulkan.is_some()
    }

    pub fn hardware_name(&self) -> Option<&str> {
        self.vulkan.as_ref().map(|v| v.device_name.as_str())
    }

    /// Allocates an uninitialized VRAM buffer of `size_bytes` on the accelerator.
    pub fn device_alloc(&mut self, name: impl Into<String>, size_bytes: usize) {
        self.device_memory.insert(name.into(), vec![0u8; size_bytes]);
    }

    /// Copies data from host RAM to GPU device memory.
    pub fn copy_to_device(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
        let buf = self
            .device_memory
            .get_mut(name)
            .ok_or_else(|| format!("GPU device memory '{}' not found", name))?;
        if buf.len() < data.len() {
            buf.resize(data.len(), 0);
        }
        buf[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Copies data from GPU device memory back to host RAM.
    pub fn copy_from_device(&self, name: &str) -> Result<Vec<u8>, String> {
        self.device_memory
            .get(name)
            .cloned()
            .ok_or_else(|| format!("GPU device memory '{}' not found", name))
    }

    /// Dispatches a compute kernel or standard Paper to run across a grid of parallel threads.
    pub fn launch_kernel(
        &mut self,
        paper: &Paper,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    ) -> Result<GpuExecutionResult, String> {
        let total_blocks = (grid_dim.0 * grid_dim.1 * grid_dim.2) as usize;
        let threads_per_block = (block_dim.0 * block_dim.1 * block_dim.2) as usize;
        let total_threads = total_blocks * threads_per_block;

        let mut last_return: Option<Value> = None;

        // Execute parallel thread iterations
        for block_idx in 0..total_blocks {
            for thread_idx in 0..threads_per_block {
                let global_thread_id = block_idx * threads_per_block + thread_idx;
                let res = self.execute_single_thread(
                    &paper.parts,
                    global_thread_id,
                    thread_idx,
                    block_idx,
                    grid_dim,
                    block_dim,
                )?;
                if res.is_some() {
                    last_return = res;
                }
            }
        }

        Ok(GpuExecutionResult {
            return_value: last_return,
            total_threads_executed: total_threads,
            grid_dim,
            block_dim,
        })
    }

    /// Executes the Paper directly on the GPU accelerator.
    pub fn execute_paper(&mut self, paper: &Paper) -> Result<GpuExecutionResult, String> {
        // First try to execute on physical Vulkan GPU hardware if available
        if let Some(ref mut vk) = self.vulkan {
            let mut grid = (1, 1, 1);
            let mut block = (1, 1, 1);
            for part in &paper.parts {
                if let Part::KernelLaunch { grid_dim, block_dim, .. } = part {
                    grid = *grid_dim;
                    block = *block_dim;
                    break;
                }
            }
            if let Ok(res) = vk.execute_paper(paper, grid, block) {
                return Ok(res);
            }
        }

        // If the paper contains explicit KernelLaunch instructions, execute them on software simulator
        let mut launched = false;
        let mut last_res = GpuExecutionResult {
            return_value: None,
            total_threads_executed: 1,
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
        };

        for part in &paper.parts {
            match part {
                Part::DeviceMemoryAlloc(name, size) => {
                    self.device_alloc(name.clone(), *size as usize);
                }
                Part::CopyToDevice(name) => {
                    // Marker in Paper body
                    let _ = self.device_memory.contains_key(name);
                }
                Part::CopyFromDevice(name) => {
                    let _ = self.device_memory.contains_key(name);
                }
                Part::KernelLaunch {
                    grid_dim,
                    block_dim,
                    body,
                    ..
                } => {
                    let sub_paper = Paper::new("GpuKernel", body.clone());
                    last_res = self.launch_kernel(&sub_paper, *grid_dim, *block_dim)?;
                    launched = true;
                }
                _ => {}
            }
        }

        if !launched {
            // Run the general paper with a single default workgroup
            last_res = self.launch_kernel(paper, (1, 1, 1), (1, 1, 1))?;
        }

        Ok(last_res)
    }

    /// Evaluates a thread's execution of Paper parts within the GPU device context.
    fn execute_single_thread(
        &mut self,
        parts: &[Part],
        global_thread_id: usize,
        thread_idx: usize,
        block_idx: usize,
        _grid_dim: (u32, u32, u32),
        _block_dim: (u32, u32, u32),
    ) -> Result<Option<Value>, String> {
        let mut stack: Vec<Value> = Vec::new();
        let mut locals: HashMap<String, Value> = HashMap::new();

        
        locals.insert("global_id".to_string(), Value::I64(global_thread_id as i64));
        locals.insert("thread_idx".to_string(), Value::I64(thread_idx as i64));
        locals.insert("block_idx".to_string(), Value::I64(block_idx as i64));

        let mut ip = 0;
        let mut labels: HashMap<String, usize> = HashMap::new();

        
        for (idx, part) in parts.iter().enumerate() {
            if let Part::Label(name) = part {
                labels.insert(name.clone(), idx);
            }
        }

        while ip < parts.len() {
            let part = &parts[ip];
            ip += 1;

            match part {
                Part::Nop | Part::Label(_) => {}
                Part::Halt => break,
                Part::Bool(b) => stack.push(Value::Bool(*b)),
                Part::String(s) => stack.push(Value::Str(std::rc::Rc::new(s.clone()))),
                Part::PushNull => stack.push(Value::Null),
                Part::Pop => {
                    stack.pop();
                }
                Part::Dup => {
                    if let Some(top) = stack.last().cloned() {
                        stack.push(top);
                    }
                }
                Part::Swap => {
                    let len = stack.len();
                    if len >= 2 {
                        stack.swap(len - 1, len - 2);
                    }
                }
                Part::RotateThree => {
                    let len = stack.len();
                    if len >= 3 {
                        let c = stack.pop().unwrap();
                        let b = stack.pop().unwrap();
                        let a = stack.pop().unwrap();
                        stack.push(b);
                        stack.push(c);
                        stack.push(a);
                    }
                }
                Part::Integer(size, val, _) => match size {
                    Size::S8 => stack.push(Value::I8(*val as i8)),
                    Size::S16 => stack.push(Value::I16(*val as i16)),
                    Size::S32 => stack.push(Value::I32(*val as i32)),
                    Size::S64 => stack.push(Value::I64(*val as i64)),
                    Size::S128 => stack.push(Value::I128(*val as i128)),
                },
                Part::Decimal(size, val) => match size {
                    Size::S32 => stack.push(Value::F32(*val as f32)),
                    _ => stack.push(Value::F64(*val)),
                },
                Part::CalculateInteger(size, op) => {
                    let rhs = stack.pop().ok_or("GPU stack underflow")?;
                    let lhs = stack.pop().ok_or("GPU stack underflow")?;
                    let to_i64 = |v: &Value| match v {
                        Value::I8(n) => Some(*n as i64),
                        Value::I16(n) => Some(*n as i64),
                        Value::I32(n) => Some(*n as i64),
                        Value::I64(n) => Some(*n),
                        Value::U8(n) => Some(*n as i64),
                        Value::U16(n) => Some(*n as i64),
                        Value::U32(n) => Some(*n as i64),
                        Value::U64(n) => Some(*n as i64),
                        _ => None,
                    };
                    let res = if let (Some(a), Some(b)) = (to_i64(&lhs), to_i64(&rhs)) {
                        let computed = match op {
                            IntegerCalculateType::Add => a.wrapping_add(b),
                            IntegerCalculateType::Sub => a.wrapping_sub(b),
                            IntegerCalculateType::Mul => a.wrapping_mul(b),
                            IntegerCalculateType::Div(false) => {
                                if b == 0 { 0 } else { a / b }
                            }
                            IntegerCalculateType::Div(true) => {
                                let (ua, ub) = (a as u64, b as u64);
                                if ub == 0 { 0 } else { (ua / ub) as i64 }
                            }
                            IntegerCalculateType::Rem(_) => {
                                if b == 0 { 0 } else { a % b }
                            }
                        };
                        match size {
                            Size::S64 => Value::I64(computed),
                            _ => Value::I32(computed as i32),
                        }
                    } else {
                        match size {
                            Size::S64 => Value::I64(0),
                            _ => Value::I32(0),
                        }
                    };
                    stack.push(res);
                }
                Part::CalculateDecimal(_size, op) => {
                    let rhs = stack.pop().ok_or("GPU stack underflow")?;
                    let lhs = stack.pop().ok_or("GPU stack underflow")?;
                    let res = match (lhs, rhs) {
                        (Value::F32(a), Value::F32(b)) => match op {
                            DecimalCalculateType::Add => Value::F32(a + b),
                            DecimalCalculateType::Sub => Value::F32(a - b),
                            DecimalCalculateType::Mul => Value::F32(a * b),
                            DecimalCalculateType::Div => Value::F32(a / b),
                            DecimalCalculateType::Rem => Value::F32(a % b),
                        },
                        (Value::F64(a), Value::F64(b)) => match op {
                            DecimalCalculateType::Add => Value::F64(a + b),
                            DecimalCalculateType::Sub => Value::F64(a - b),
                            DecimalCalculateType::Mul => Value::F64(a * b),
                            DecimalCalculateType::Div => Value::F64(a / b),
                            DecimalCalculateType::Rem => Value::F64(a % b),
                        },
                        _ => Value::F64(0.0),
                    };
                    stack.push(res);
                }
                Part::DecimalNegative => {
                    if let Some(val) = stack.pop() {
                        match val {
                            Value::F32(f) => stack.push(Value::F32(-f)),
                            Value::F64(f) => stack.push(Value::F64(-f)),
                            _ => stack.push(val),
                        }
                    }
                }
                Part::ComparisonInteger(op, _unsigned, _) => {
                    let rhs = stack.pop().ok_or("GPU stack underflow")?;
                    let lhs = stack.pop().ok_or("GPU stack underflow")?;
                    let cmp = match (lhs, rhs) {
                        (Value::I32(a), Value::I32(b)) => match op {
                            IntegerComparisonType::Equal => a == b,
                            IntegerComparisonType::NotEqual => a != b,
                            IntegerComparisonType::Greater => a > b,
                            IntegerComparisonType::GreaterEqual => a >= b,
                            IntegerComparisonType::Less => a < b,
                            IntegerComparisonType::LessEqual => a <= b,
                        },
                        (Value::I64(a), Value::I64(b)) => match op {
                            IntegerComparisonType::Equal => a == b,
                            IntegerComparisonType::NotEqual => a != b,
                            IntegerComparisonType::Greater => a > b,
                            IntegerComparisonType::GreaterEqual => a >= b,
                            IntegerComparisonType::Less => a < b,
                            IntegerComparisonType::LessEqual => a <= b,
                        },
                        _ => false,
                    };
                    stack.push(Value::Bool(cmp));
                }
                Part::ComparisonDecimal(op, _) => {
                    let rhs = stack.pop().ok_or("GPU stack underflow")?;
                    let lhs = stack.pop().ok_or("GPU stack underflow")?;
                    let cmp = match (lhs, rhs) {
                        (Value::F32(a), Value::F32(b)) => match op {
                            DecimalComparisonType::Equal => a == b,
                            DecimalComparisonType::NotEqual => a != b,
                            DecimalComparisonType::Greater => a > b,
                            DecimalComparisonType::GreaterEqual => a >= b,
                            DecimalComparisonType::Less => a < b,
                            DecimalComparisonType::LessEqual => a <= b,
                        },
                        (Value::F64(a), Value::F64(b)) => match op {
                            DecimalComparisonType::Equal => a == b,
                            DecimalComparisonType::NotEqual => a != b,
                            DecimalComparisonType::Greater => a > b,
                            DecimalComparisonType::GreaterEqual => a >= b,
                            DecimalComparisonType::Less => a < b,
                            DecimalComparisonType::LessEqual => a <= b,
                        },
                        _ => false,
                    };
                    stack.push(Value::Bool(cmp));
                }
                Part::BinaryOperation(op) => {
                    let rhs = stack.pop().ok_or("GPU stack underflow")?;
                    let lhs = stack.pop().ok_or("GPU stack underflow")?;
                    let res = match (lhs, rhs) {
                        (Value::I32(a), Value::I32(b)) => match op {
                            BinaryOperationType::LShift => Value::I32(a << (b & 31)),
                            BinaryOperationType::RShift(false) => Value::I32((a as u32 >> (b & 31)) as i32),
                            BinaryOperationType::RShift(true) => Value::I32(a >> (b & 31)),
                            BinaryOperationType::And => Value::I32(a & b),
                            BinaryOperationType::Or => Value::I32(a | b),
                            BinaryOperationType::Xor => Value::I32(a ^ b),
                        },
                        (Value::I64(a), Value::I64(b)) => match op {
                            BinaryOperationType::LShift => Value::I64(a << (b & 63)),
                            BinaryOperationType::RShift(false) => Value::I64((a as u64 >> (b & 63)) as i64),
                            BinaryOperationType::RShift(true) => Value::I64(a >> (b & 63)),
                            BinaryOperationType::And => Value::I64(a & b),
                            BinaryOperationType::Or => Value::I64(a | b),
                            BinaryOperationType::Xor => Value::I64(a ^ b),
                        },
                        _ => Value::I32(0),
                    };
                    stack.push(res);
                }
                Part::BitwiseNot(_) => {
                    if let Some(val) = stack.pop() {
                        match val {
                            Value::I32(i) => stack.push(Value::I32(!i)),
                            Value::I64(i) => stack.push(Value::I64(!i)),
                            _ => stack.push(val),
                        }
                    }
                }
                Part::FusedMultiplyAdd(_) => {
                    let c = stack.pop().ok_or("GPU stack underflow")?;
                    let b = stack.pop().ok_or("GPU stack underflow")?;
                    let a = stack.pop().ok_or("GPU stack underflow")?;
                    match (a, b, c) {
                        (Value::F32(va), Value::F32(vb), Value::F32(vc)) => {
                            stack.push(Value::F32(va.mul_add(vb, vc)));
                        }
                        (Value::F64(va), Value::F64(vb), Value::F64(vc)) => {
                            stack.push(Value::F64(va.mul_add(vb, vc)));
                        }
                        _ => stack.push(Value::F64(0.0)),
                    }
                }
                Part::UnaryMath(_size, math_type) => {
                    let val = stack.pop().ok_or("GPU stack underflow")?;
                    match math_type {
                        UnaryMathType::Abs => match val {
                            Value::I32(i) => stack.push(Value::I32(i.abs())),
                            Value::I64(i) => stack.push(Value::I64(i.abs())),
                            Value::F32(f) => stack.push(Value::F32(f.abs())),
                            Value::F64(f) => stack.push(Value::F64(f.abs())),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Sqrt => match val {
                            Value::F32(f) => stack.push(Value::F32(f.sqrt())),
                            Value::F64(f) => stack.push(Value::F64(f.sqrt())),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Increment => match val {
                            Value::I32(i) => stack.push(Value::I32(i + 1)),
                            Value::I64(i) => stack.push(Value::I64(i + 1)),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Decrement => match val {
                            Value::I32(i) => stack.push(Value::I32(i - 1)),
                            Value::I64(i) => stack.push(Value::I64(i - 1)),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Negate => match val {
                            Value::I32(i) => stack.push(Value::I32(-i)),
                            Value::I64(i) => stack.push(Value::I64(-i)),
                            Value::F32(f) => stack.push(Value::F32(-f)),
                            Value::F64(f) => stack.push(Value::F64(-f)),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Floor => match val {
                            Value::F32(f) => stack.push(Value::F32(f.floor())),
                            Value::F64(f) => stack.push(Value::F64(f.floor())),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Ceil => match val {
                            Value::F32(f) => stack.push(Value::F32(f.ceil())),
                            Value::F64(f) => stack.push(Value::F64(f.ceil())),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Round => match val {
                            Value::F32(f) => stack.push(Value::F32(f.round())),
                            Value::F64(f) => stack.push(Value::F64(f.round())),
                            _ => stack.push(val),
                        },
                        UnaryMathType::Truncate => match val {
                            Value::F32(f) => stack.push(Value::F32(f.trunc())),
                            Value::F64(f) => stack.push(Value::F64(f.trunc())),
                            _ => stack.push(val),
                        },
                    }
                }
                Part::ReturnInteger => {
                    return Ok(stack.pop());
                }
                Part::ReturnVoid => {
                    return Ok(None);
                }
                Part::PushLocal(name) => {
                    let val = locals.get(name).cloned().unwrap_or(Value::Null);
                    stack.push(val);
                }
                Part::StoreLocal(name) => {
                    if let Some(val) = stack.pop() {
                        locals.insert(name.clone(), val);
                    }
                }
                Part::Jump(label) => {
                    if let Some(target) = labels.get(label) {
                        ip = *target;
                    }
                }
                Part::JumpIfTrue(label) => {
                    let cond = stack.pop().map_or(false, |v| v.is_truthy());
                    if cond {
                        if let Some(target) = labels.get(label) {
                            ip = *target;
                        }
                    }
                }
                Part::JumpIfFalse(label) => {
                    let cond = stack.pop().map_or(false, |v| v.is_truthy());
                    if !cond {
                        if let Some(target) = labels.get(label) {
                            ip = *target;
                        }
                    }
                }
                
                Part::AllocStackStruct(_name) | Part::AllocStruct(_name) => {
                    stack.push(Value::RawMemory(Rc::new(RefCell::new(vec![0u8; 64]))));
                }
                Part::SetStructField { offset, size, .. } => {
                    let val = stack.pop().ok_or("GPU struct underflow")?;
                    let target = stack.last().ok_or("GPU struct underflow")?;
                    if let Value::RawMemory(mem) = target {
                        let mut bytes = mem.borrow_mut();
                        let sz = *size as usize / 8;
                        if bytes.len() < offset + sz {
                            bytes.resize(offset + sz, 0);
                        }
                        match val {
                            Value::I32(v) => {
                                bytes[*offset..*offset + 4].copy_from_slice(&v.to_le_bytes());
                            }
                            Value::I64(v) => {
                                bytes[*offset..*offset + 8].copy_from_slice(&v.to_le_bytes());
                            }
                            Value::F32(v) => {
                                bytes[*offset..*offset + 4].copy_from_slice(&v.to_le_bytes());
                            }
                            Value::F64(v) => {
                                bytes[*offset..*offset + 8].copy_from_slice(&v.to_le_bytes());
                            }
                            _ => {}
                        }
                    }
                }
                Part::GetStructField { offset, size, .. } => {
                    let target = stack.pop().ok_or("GPU struct underflow")?;
                    if let Value::RawMemory(mem) = target {
                        let bytes = mem.borrow();
                        match size {
                            Size::S32 => {
                                let slice = &bytes[*offset..*offset + 4];
                                let v = i32::from_le_bytes(slice.try_into().unwrap());
                                stack.push(Value::I32(v));
                            }
                            Size::S64 => {
                                let slice = &bytes[*offset..*offset + 8];
                                let v = i64::from_le_bytes(slice.try_into().unwrap());
                                stack.push(Value::I64(v));
                            }
                            _ => stack.push(Value::I32(0)),
                        }
                    }
                }
                Part::GetStructFieldPtr { offset, .. } => {
                    stack.push(Value::Pointer(*offset));
                }
                Part::DeviceMemoryAlloc(name, size) => {
                    self.device_alloc(name.clone(), *size as usize);
                }
                Part::Print => {
                    if let Some(top) = stack.pop() {
                        match top {
                            Value::Str(ref s) => println!("{}", s),
                            Value::I32(v) => println!("{}", v),
                            Value::I64(v) => println!("{}", v),
                            _ => println!("{:?}", top),
                        }
                    }
                }
                Part::SpawnDetached(name) => {
                    if let Some(paper) = crate::vm::pool::get_global_paper(name) {
                        crate::vm::pool::IrisThreadPool::global().spawn(move || {
                            let mut vm = crate::vm::engine::IrisEngine::new();
                            let _ = paper.run_vm(&mut vm);
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(stack.pop())
    }
}
