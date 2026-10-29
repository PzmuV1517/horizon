//! A shader pack's custom textures (`texture.<stage>.<name>`, `texture.noise`) on the GPU, and
//! which of them a program's sampler reads.

use horizon_shaders::{CustomTexture, Group, Kind, Texels, Texture, TextureDim, TextureSource};

/// What a custom texture is once uploaded: an image of ours, or one the game already has.
pub(super) enum Image {
    View(wgpu::TextureView),
    Lightmap,
    Atlas,
}

pub(super) struct PackTexture {
    pub(super) stage: String,
    pub(super) name: String,
    pub(super) dimension: wgpu::TextureViewDimension,
    pub(super) sample_type: wgpu::TextureSampleType,
    pub(super) blur: bool,
    pub(super) clamp: bool,
    pub(super) image: Image,
}

/// Uploads `textures`, with a note for each the game can't provide.
pub(super) fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    textures: Vec<CustomTexture>,
    notes: &mut Vec<String>,
) -> Vec<PackTexture> {
    let features = device.features();
    let mut out = Vec::new();
    for t in textures {
        let (image, dimension, sample_type, blur, clamp) = match t.source {
            TextureSource::Image(texture) => {
                let (view, sample_type) = create(device, queue, &texture, features);
                let dimension = match texture.dim {
                    TextureDim::D1 => wgpu::TextureViewDimension::D1,
                    TextureDim::D2 => wgpu::TextureViewDimension::D2,
                    TextureDim::D3 => wgpu::TextureViewDimension::D3,
                };
                (
                    Image::View(view),
                    dimension,
                    sample_type,
                    texture.blur,
                    texture.clamp,
                )
            }
            TextureSource::Resource(location) => {
                let path = location.strip_prefix("minecraft:").unwrap_or(&location);
                let image = if path.starts_with("dynamic/lightmap") {
                    Image::Lightmap
                } else if path.starts_with("textures/atlas/blocks") || path == "dynamic/atlas" {
                    Image::Atlas
                } else {
                    notes.push(format!(
                        "texture.{}.{}: `{location}` is not provided yet",
                        t.stage, t.name
                    ));
                    continue;
                };
                let float = wgpu::TextureSampleType::Float { filterable: true };
                (image, wgpu::TextureViewDimension::D2, float, true, true)
            }
        };
        out.push(PackTexture {
            stage: t.stage,
            name: t.name,
            dimension,
            sample_type,
            blur,
            clamp,
            image,
        });
    }
    out
}

/// The stage name of `texture.<stage>.*` a program of `group` reads.
fn stage(group: Group) -> &'static str {
    match group {
        Group::Gbuffers | Group::Translucent | Group::Shadow => "gbuffers",
        Group::Prepare => "prepare",
        Group::Deferred => "deferred",
        Group::Composite | Group::Final => "composite",
        Group::ShadowComp => "shadowcomp",
    }
}

/// The custom texture a `ty name` sampler of a `group` program reads: same stage, the same
/// name (or another name of the same buffer, `gaux1` for `colortex4`) and dimension. A unit
/// may hold one texture per dimension.
pub(super) fn find(textures: &[PackTexture], group: Group, ty: &str, name: &str) -> Option<usize> {
    let dimension = if ty.contains("1D") {
        wgpu::TextureViewDimension::D1
    } else if ty.contains("3D") {
        wgpu::TextureViewDimension::D3
    } else {
        wgpu::TextureViewDimension::D2
    };
    let buffer = horizon_shaders::buffer_index(name);
    textures.iter().position(|t| {
        t.stage == stage(group)
            && t.dimension == dimension
            && (t.name == name
                || buffer.is_some() && horizon_shaders::buffer_index(&t.name) == buffer)
    })
}

/// `texture.noise`, if the pack has one.
pub(super) fn noise(textures: &[PackTexture]) -> Option<&wgpu::TextureView> {
    textures
        .iter()
        .find(|t| t.stage == "noise" && t.dimension == wgpu::TextureViewDimension::D2)
        .and_then(|t| match &t.image {
            Image::View(v) => Some(v),
            _ => None,
        })
}

/// The wgpu format a texture of `texture.format` is stored in: sampled, never rendered to, so
/// any format whose texels hold the values.
fn format(texture: &Texture) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as T;
    let f = texture.format;
    let pick = |r, rg, rgba| match texture.components() {
        1 => r,
        2 => rg,
        _ => rgba,
    };
    match (f.kind, f.bits) {
        (Kind::Unorm, b) if b <= 8 => pick(T::R8Unorm, T::Rg8Unorm, T::Rgba8Unorm),
        (Kind::Snorm, 8) => pick(T::R8Snorm, T::Rg8Snorm, T::Rgba8Snorm),
        (Kind::Float, 32) => pick(T::R32Float, T::Rg32Float, T::Rgba32Float),
        (Kind::Int, 8) => pick(T::R8Sint, T::Rg8Sint, T::Rgba8Sint),
        (Kind::Int, 16) => pick(T::R16Sint, T::Rg16Sint, T::Rgba16Sint),
        (Kind::Int, _) => pick(T::R32Sint, T::Rg32Sint, T::Rgba32Sint),
        (Kind::Uint, 8) => pick(T::R8Uint, T::Rg8Uint, T::Rgba8Uint),
        (Kind::Uint, 16) => pick(T::R16Uint, T::Rg16Uint, T::Rgba16Uint),
        (Kind::Uint, _) => pick(T::R32Uint, T::Rg32Uint, T::Rgba32Uint),
        // 16-bit and packed normalized formats, half floats and the small floats: halves hold
        // them closely enough.
        _ => pick(T::R16Float, T::Rg16Float, T::Rgba16Float),
    }
}

fn create(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &Texture,
    features: wgpu::Features,
) -> (wgpu::TextureView, wgpu::TextureSampleType) {
    let format = format(texture);
    let bytes: Vec<u8> = match (&texture.texels, format) {
        (Texels::Float(v), f) if f.block_copy_size(None) == Some(texture.components() as u32) => {
            if matches!(
                f,
                wgpu::TextureFormat::R8Snorm
                    | wgpu::TextureFormat::Rg8Snorm
                    | wgpu::TextureFormat::Rgba8Snorm
            ) {
                v.iter()
                    .map(|x| (x.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8)
                    .collect()
            } else {
                v.iter()
                    .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                    .collect()
            }
        }
        (Texels::Float(v), f)
            if f.block_copy_size(None) == Some(4 * texture.components() as u32) =>
        {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        }
        (Texels::Float(v), _) => v
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect(),
        (Texels::Int(v), f) => {
            let width = f.block_copy_size(None).unwrap_or(4) as usize / texture.components();
            v.iter()
                .flat_map(|x| x.to_le_bytes().into_iter().take(width))
                .collect()
        }
        (Texels::Uint(v), f) => {
            let width = f.block_copy_size(None).unwrap_or(4) as usize / texture.components();
            v.iter()
                .flat_map(|x| x.to_le_bytes().into_iter().take(width))
                .collect()
        }
    };
    let [width, height, depth] = texture.size;
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: depth,
    };
    let dimension = match texture.dim {
        TextureDim::D1 => wgpu::TextureDimension::D1,
        TextureDim::D2 => wgpu::TextureDimension::D2,
        TextureDim::D3 => wgpu::TextureDimension::D3,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Shader pack custom texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let row = width * format.block_copy_size(None).unwrap_or(4);
    queue.write_texture(
        tex.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: Some(height),
        },
        size,
    );
    let sample_type = format
        .sample_type(None, Some(features))
        .unwrap_or(wgpu::TextureSampleType::Float { filterable: false });
    (tex.create_view(&Default::default()), sample_type)
}
