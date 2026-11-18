use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Part {
    
    Print,
    Bool(bool),
    String(String),
    PushNull,
    Pop,
    Dup,
    Swap,
    RotateThree,
    Pick(usize),
    Roll(usize),
    Drop(usize),
    DupMultiple(usize),
    Nop,
    Halt,

    
    Integer(Size, u64, bool),
    CalculateInteger(Size, IntegerCalculateType),
    ComparisonInteger(IntegerComparisonType, bool, bool),
    ResizeInteger(Size, Size, bool),

    
    Decimal(Size, f64),
    CalculateDecimal(Size, DecimalCalculateType),
    DecimalNegative,
    ComparisonDecimal(DecimalComparisonType, bool),
    ResizeDecimal(bool),

    
    UnaryMath(Size, UnaryMathType),
    FusedMultiplyAdd(Size),

    
    IntegerToDecimal(bool, Size),
    DecimalToInteger(bool, Size),
    Bitcast(Size, Size),

    
    BinaryOperation(BinaryOperationType),
    BitwiseNot(Size),

    
    ReturnInteger,
    ReturnVoid,
    PushLocal(String),
    StoreLocal(String),
    PushGlobal(String),
    StoreGlobal(String),
    Jump(String),
    JumpIfTrue(String),
    JumpIfFalse(String),
    JumpIfNull(String),
    JumpIfNonNull(String),
    TableSwitch {
        default: String,
        cases: Vec<(i64, String)>,
    },
    Call(String, Vec<Part>),
    TailCall(String, Vec<Part>),
    Label(String),

    
    DefineStruct {
        name: String,
        size: usize,
        align: usize,
        fields: Vec<StructField>,
    },
    
    AllocStackStruct(String),
    
    AllocStruct(String),
    
    GetStructField {
        struct_name: String,
        field_index: usize,
        offset: usize,
        size: Size,
    },
    
    SetStructField {
        struct_name: String,
        field_index: usize,
        offset: usize,
        size: Size,
    },
    
    GetStructFieldPtr {
        struct_name: String,
        field_index: usize,
        offset: usize,
    },
    
    
    CallMethod {
        struct_name: String,
        method_name: String,
        args: Vec<Part>,
    },

    
    AllocRaw {
        size: usize,
        align: usize,
    },
    FreeRaw,
    LoadPtr(Size),
    StorePtr(Size),
    OffsetPtr(i64),
    MemCopy {
        size: usize,
    },
    MemSet {
        value: u8,
        size: usize,
    },

    
    CreateArray {
        element_type: Option<String>,
        size: usize,
    },
    GetArrayElement,
    SetArrayElement,
    GetArrayLength,
    ResizeArray,
    CreateMap {
        capacity: usize,
    },
    MapGet,
    MapSet,
    MapContainsKey,
    MapRemoveKey,
    MapGetOrDefault,
    AllocateSlice,

    
    AtomicAdd(Size),
    AtomicSub(Size),
    AtomicCas(Size),
    EnterMonitor,
    ExitMonitor,
    YieldThread,
    
    SpawnDetached(String),

    
    CallWithCache {
        method: String,
        args: Vec<Part>,
    },
    GetPropertyWithCache(String),
    SetPropertyWithCache(String),
    MegamorphicCall {
        method: String,
        args: Vec<Part>,
    },

    
    Throw,
    Try(String), 
    EndTry,

    
    InvokeNative(String, Vec<Part>),

    
    KernelLaunch {
        name: String,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        body: Vec<Part>,
    },
    DeviceMemoryAlloc(String, Size),
    CopyToDevice(String),
    CopyFromDevice(String),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub ty: FieldType,
    pub offset: usize,
}

impl StructField {
    pub fn new(name: impl Into<String>, ty: FieldType, offset: usize) -> Self {
        Self {
            name: name.into(),
            ty,
            offset,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum FieldType {
    Integer(Size),
    Decimal(Size),
    Bool,
    Pointer,
    NestedStruct(String),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Size {
    S8 = 8,
    S16 = 16,
    S32 = 32,
    S64 = 64,
    S128 = 128,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum UnaryMathType {
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Round,
    Truncate,
    Negate,
    Increment,
    Decrement,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum IntegerCalculateType {
    Add,
    Sub,
    Mul,
    Div(bool),
    Rem(bool),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum IntegerComparisonType {
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum DecimalCalculateType {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum DecimalComparisonType {
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperationType {
    LShift,
    RShift(bool),
    And,
    Or,
    Xor,
}
