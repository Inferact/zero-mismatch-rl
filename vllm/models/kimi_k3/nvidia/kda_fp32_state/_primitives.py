"""Packed FP32 arithmetic and asynchronous copies via explicit PTX."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


def unpack(value, loc, ip):
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), value, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@dsl_user_op
def mul_packed_f32x2(a, b, *, rnd="rn", loc=None, ip=None):
    assert rnd == "rn"
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32,f32)>"),
        [x.ir_value(loc=loc, ip=ip) for x in (*a, *b)],
        "{ .reg .b64 a,b,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
        "mul.rn.f32x2 d,a,b; mov.b64 {$0,$1},d; }",
        "=f,=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return unpack(result, loc, ip)


@dsl_user_op
def add_packed_f32x2(a, b, *, rnd="rn", loc=None, ip=None):
    assert rnd == "rn"
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32,f32)>"),
        [x.ir_value(loc=loc, ip=ip) for x in (*a, *b)],
        "{ .reg .b64 a,b,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
        "add.rn.f32x2 d,a,b; mov.b64 {$0,$1},d; }",
        "=f,=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return unpack(result, loc, ip)


@dsl_user_op
def fma_packed_f32x2(a, b, c, *, rnd="rn", loc=None, ip=None):
    assert rnd == "rn"
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32,f32)>"),
        [x.ir_value(loc=loc, ip=ip) for x in (*a, *b, *c)],
        "{ .reg .b64 a,b,c,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
        "mov.b64 c,{$6,$7}; fma.rn.f32x2 d,a,b,c; mov.b64 {$0,$1},d; }",
        "=f,=f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return unpack(result, loc, ip)


@dsl_user_op
def elected_bulk_copy_g2s(dst, src_address, size, mbar, *, loc=None, ip=None):
    """bulk_copy_g2s issued by one elected lane of a converged warp (elect.sync
    inside the asm), so the warp needs no per-lane issue loop."""
    dst_i32 = llvm.ptrtoint(T.i32(), dst.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    mbar_i32 = llvm.ptrtoint(T.i32(), mbar.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            dst_i32,
            src_address.ir_value(loc=loc, ip=ip),
            size.ir_value(loc=loc, ip=ip),
            mbar_i32,
        ],
        "{ .reg .pred p; .reg .b32 r; elect.sync r|p, 0xffffffff; "
        "@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3]; }",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def elected_arrive_expect_tx(mbar, size, *, loc=None, ip=None):
    """mbarrier arrive + expect_tx by one elected lane of a converged warp."""
    mbar_i32 = llvm.ptrtoint(T.i32(), mbar.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [mbar_i32, size.ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; .reg .b32 r; elect.sync r|p, 0xffffffff; "
        "@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1; }",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def evict_last_policy(*, loc=None, ip=None):
    """An L2 evict_last cache policy, created once and reused by stores."""
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store4_policy(address, values, policy, *, loc=None, ip=None):
    """16-byte global store of four FP32 values under a given L2 cache policy
    (placement only, same bits)."""
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)]
        + [v.ir_value(loc=loc, ip=ip) for v in values]
        + [policy.ir_value(loc=loc, ip=ip)],
        "st.global.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, $5;",
        "l,f,f,f,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _known_aligned(pointer, alignment: cutlass.Constexpr[int]):
    return cute.make_ptr(
        pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=alignment
    )
