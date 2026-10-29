//! A shader pack's own images (`image.<name>`) and storage buffers (`bufferObject.<n>`) on the
//! GPU: created at the render size they depend on, kept across frames, images cleared before a
//! frame when the pack asks.

use std::collections::BTreeMap;

use horizon_shaders::{BufferObject, CustomImage, Format, Kind};

pub(super) struct PackImage {
    pub(super) def: CustomImage,
    pub(super) format: wgpu::TextureFormat,
    pub(super) dimension: wgpu::TextureViewDimension,
    pub(super) view: Option<wgpu::TextureView>,
    /// The render size the image was made for (relative images follow it).
    made_for: [u32; 2],
}

impl PackImage {
    pub(super) fn sample_type(&self, features: wgpu::Features) -> wgpu::TextureSampleType {
        self.format
            .sample_type(None, Some(features))
            .unwrap_or(wgpu::TextureSampleType::Float { filterable: false })
    }
}

/// The images `image.*` declares.
pub(super) fn images(defs: Vec<CustomImage>) -> Vec<PackImage> {
    defs.into_iter()
        .map(|def| PackImage {
            format: format(def.format),
            dimension: if def.is_3d() {
                wgpu::TextureViewDimension::D3
            } else {
                wgpu::TextureViewDimension::D2
            },
            def,
            view: None,
            made_for: [0; 2],
        })
        .collect()
}

/// The wgpu format of an image's internal format: a storage format of the same channels.
fn format(f: Format) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as T;
    let pick = |r, rg, rgba| match f.channels {
        1 => r,
        2 => rg,
        _ => rgba,
    };
    match (f.kind, f.bits) {
        (Kind::Int, 8) => pick(T::R8Sint, T::Rg8Sint, T::Rgba8Sint),
        (Kind::Int, 16) => pick(T::R16Sint, T::Rg16Sint, T::Rgba16Sint),
        (Kind::Int, _) => pick(T::R32Sint, T::Rg32Sint, T::Rgba32Sint),
        (Kind::Uint, 8) => pick(T::R8Uint, T::Rg8Uint, T::Rgba8Uint),
        (Kind::Uint, 16) => pick(T::R16Uint, T::Rg16Uint, T::Rgba16Uint),
        (Kind::Uint, _) => pick(T::R32Uint, T::Rg32Uint, T::Rgba32Uint),
        (Kind::Float, 32) => pick(T::R32Float, T::Rg32Float, T::Rgba32Float),
        (Kind::Float, _) => pick(T::R16Float, T::Rg16Float, T::Rgba16Float),
        (Kind::Snorm, _) => pick(T::R8Snorm, T::Rg8Snorm, T::Rgba8Snorm),
        (Kind::Unorm, 16) => pick(T::R16Unorm, T::Rg16Unorm, T::Rgba16Unorm),
        (Kind::Unorm, _) => pick(T::R8Unorm, T::Rg8Unorm, T::Rgba8Unorm),
    }
}

/// Makes the images that are missing or sized for another render size.
pub(super) fn resize_images(device: &wgpu::Device, images: &mut [PackImage], size: [u32; 2]) {
    for image in images {
        let relative = matches!(image.def.size, horizon_shaders::ImageSize::Relative(_));
        if image.view.is_some() && (!relative || image.made_for == size) {
            continue;
        }
        let [width, height, depth] = image.def.extent(size);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&image.def.name),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: if image.dimension == wgpu::TextureViewDimension::D3 {
                wgpu::TextureDimension::D3
            } else {
                wgpu::TextureDimension::D2
            },
            format: image.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        image.view = Some(texture.create_view(&Default::default()));
        image.made_for = size;
    }
}

/// Clears the images the pack clears before each frame.
pub(super) fn clear(encoder: &mut wgpu::CommandEncoder, images: &[PackImage]) {
    for view in images
        .iter()
        .filter(|i| i.def.clear)
        .filter_map(|i| i.view.as_ref())
    {
        encoder.clear_texture(view.texture(), &Default::default());
    }
}

/// The pack's storage buffers, bound in set 4.
pub(super) struct Buffers {
    pub(super) layout: wgpu::BindGroupLayout,
    objects: BTreeMap<u32, BufferObject>,
    pub(super) group: Option<wgpu::BindGroup>,
    buffers: Vec<wgpu::Buffer>,
    made_for: [u32; 2],
}

impl Buffers {
    pub(super) fn new(device: &wgpu::Device, objects: BTreeMap<u32, BufferObject>) -> Self {
        let entries: Vec<_> = objects
            .keys()
            .map(|&binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shader pack buffers"),
            entries: &entries,
        });
        Self {
            layout,
            objects,
            group: None,
            buffers: Vec::new(),
            made_for: [0; 2],
        }
    }

    /// Makes the buffers (again, when one depends on the render size and it changed): their
    /// contents start zeroed and persist across frames.
    pub(super) fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        let relative = self
            .objects
            .values()
            .any(|o| matches!(o, BufferObject::PerPixel { .. }));
        if self.group.is_some() && (!relative || self.made_for == size) {
            return;
        }
        self.buffers = self
            .objects
            .values()
            .map(|o| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Shader pack buffer"),
                    size: o.size(size).next_multiple_of(4),
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let entries: Vec<_> = self
            .objects
            .keys()
            .zip(&self.buffers)
            .map(|(&binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shader pack buffers"),
            layout: &self.layout,
            entries: &entries,
        }));
        self.made_for = size;
    }

    /// Whether the pack declares buffer `binding`.
    pub(super) fn has(&self, binding: u32) -> bool {
        self.objects.contains_key(&binding)
    }
}
