struct BasicResidualBlock{C1,C2,GN1,GN2,GN3,DO,DR}
    conv1::C1
    conv2::C2
    gn1::GN1
    gn2::GN2
    gn3::GN3
    downsample::DO
    dropout::DR
end

Flux.@functor BasicResidualBlock

Flux.gpu(b::BasicResidualBlock) = BasicResidualBlock(
    Flux.gpu(b.conv1),
    Flux.gpu(b.conv2),
    Flux.gpu(b.gn1),
    Flux.gpu(b.gn2),
    Flux.gpu(b.gn3),
    Flux.gpu(b.downsample),
    Flux.gpu(b.dropout),
)

Flux.cpu(b::BasicResidualBlock) = BasicResidualBlock(
    Flux.cpu(b.conv1),
    Flux.cpu(b.conv2),
    Flux.cpu(b.gn1),
    Flux.cpu(b.gn2),
    Flux.cpu(b.gn3),
    Flux.cpu(b.downsample),
    Flux.cpu(b.dropout),
)

function BasicResidualBlock(
    outdims::Tuple,
    inplanes::Int,
    planes::Int;
    deq_expand::Int = 5,
    num_gn_groups::Int = 4,
    downsample = identity,
    n_big_kernels::Int = 0,
    dropout_rate::Real = 0.0f0,
    gn_affine::Bool = true,
)
    inner_planes = planes * deq_expand
    conv1 = Conv(
        n_big_kernels >= 1 ? (5, 5) : (3, 3),
        inplanes => inner_planes;
        stride = 1,
        pad = n_big_kernels >= 1 ? 2 : 1,
        bias = false,
    )
    gn1 = GroupNormV2(inner_planes, num_gn_groups, relu; affine = gn_affine, track_stats = true)

    conv2 = Conv(
        n_big_kernels >= 1 ? (5, 5) : (3, 3),
        inner_planes => planes;
        stride = 1,
        pad = n_big_kernels >= 1 ? 2 : 1,
        bias = false,
    )
    gn2 = GroupNormV2(planes, num_gn_groups, relu; affine = gn_affine, track_stats = true)

    gn3 = GroupNormV2(planes, num_gn_groups; affine = gn_affine, track_stats = true)

    return BasicResidualBlock(
        conv1,
        conv2,
        gn1,
        gn2,
        gn3,
        downsample,
        iszero(dropout_rate) ? identity : VariationalHiddenDropout(dropout_rate, (outdims..., planes)),
    )
end

function (b::BasicResidualBlock)(
    x::AbstractArray{T},
    injection::Union{AbstractArray{T},T} = T(0),
) where {T}
    # WTF!!! Conv is not type stable
    x_ = x
    x = b.conv1(x) :: typeof(x)
    x = b.conv2(b.gn1(x)) :: typeof(x)
    residual = b.downsample(x_) :: typeof(x)
    return b.gn3(relu.(b.gn2(b.dropout(x) .+ injection) .+ residual))
end

function Base.show(io::IO, l::BasicResidualBlock)
    p, _ = Flux.destructure(l)
    print(
        io,
        string(typeof(l).name.name),
        "() ",
        string(length(p)),
        " Trainable Parameters",
    )
end
