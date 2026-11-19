"""Aligned FP32 state transfers on SM100/SM103."""

import cutlass
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


@dsl_user_op
def load256(pointer, *, loc=None, ip=None):
    address = llvm.ptrtoint(
        T.i64(), pointer.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip
    )
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32,f32,f32,f32,f32,f32,f32,f32)>"),
        [address],
        "ld.global.v8.f32 {$0,$1,$2,$3,$4,$5,$6,$7},[$8];",
        "=f,=f,=f,=f,=f,=f,=f,=f,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(8)
    )


@dsl_user_op
def store256(pointer, values, *, loc=None, ip=None):
    address = llvm.ptrtoint(
        T.i64(), pointer.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip
    )
    llvm.inline_asm(
        None,
        [address] + [x.ir_value(loc=loc, ip=ip) for x in values],
        "st.global.v8.f32 [$0],{$1,$2,$3,$4,$5,$6,$7,$8};",
        "l,f,f,f,f,f,f,f,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_bf16_if(address, value, pred, *, loc=None, ip=None):
    """RN BF16 store of an FP32 value, predicated on pred != 0 (no branch)."""
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
            pred.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; .reg .b16 r; setp.ne.s32 p, $2, 0; "
        "cvt.rn.bf16.f32 r, $1; @p st.global.b16 [$0], r; }",
        "l,f,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
