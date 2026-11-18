use inkwell::execution_engine::ExecutionEngine;
use crate::vm::value::Value;
use crate::vm::engine::IrisEngine;
use serde::{Serialize, Deserialize};
use crate::builder::compiler::RuntimeFunctionCompiler;
use crate::builder::paper::Paper;

#[derive(Debug, Serialize, Deserialize)]
pub enum FunctionKind {
    Paper,
    Bytecode,
    Native,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub kind: FunctionKind,
    pub arity: usize,
    pub paper: Option<Paper>,
    pub bytecode: Option<Vec<u8>>,
    pub constants: Vec<Value>,
    #[serde(skip)]
    pub native: Option<unsafe extern "C" fn(iris_engine: *mut IrisEngine)>
}

impl Function {
    pub fn new_paper(name: String, arity: usize, paper: Option<Paper>) -> Self {
        Self {
            name,
            kind: FunctionKind::Paper,
            arity,
            paper,
            bytecode: None,
            constants: Vec::new(),
            native: None,
        }
    }

    pub fn new_bytecode(name: String, arity: usize, bytecode: Vec<u8>, constants: Vec<Value>) -> Self {
        Self {
            name,
            kind: FunctionKind::Bytecode,
            arity,
            paper: None,
            bytecode: Some(bytecode),
            constants,
            native: None
        }
    }

    pub fn new_native(name: String, arity: usize, native: unsafe extern "C" fn(iris_engine: *mut IrisEngine)) -> Self {
        Self {
            name,
            kind: FunctionKind::Native,
            arity,
            paper: None,
            bytecode: None,
            constants: Vec::new(),
            native: Some(native)
        }
    }

    pub fn constants(&self) -> &Vec<Value> {
        &self.constants
    }

    pub fn switch_bytecode(&mut self, bytecode: Vec<u8>, constants: Vec<Value>) {
        self.bytecode = Some(bytecode);
        self.constants = constants;
        self.kind = FunctionKind::Bytecode;
    }

    pub fn switch_native(&mut self, native: unsafe extern "C" fn(iris_engine: *mut IrisEngine)){
        self.native = Some(native);
        self.kind = FunctionKind::Native;
    }

    pub fn push_bytecode(&mut self, rfc: &RuntimeFunctionCompiler) {
        if self.paper.is_some() {
            let chunk = rfc.ivm_compile(self.paper.clone().unwrap()).unwrap();
            self.switch_bytecode(chunk.code, self.constants.to_vec());
        }
    }

    pub fn push_native<'a>(&mut self, rfc: &RuntimeFunctionCompiler<'a>, ee: ExecutionEngine<'a>) {
        let func = rfc.jit_compile(self.paper.clone().unwrap(), &ee).unwrap();
        unsafe {
            let raw_fn = func.as_raw() as unsafe extern "C" fn(iris_engine: *mut IrisEngine);
            self.switch_native(raw_fn);
        }
    }
}
