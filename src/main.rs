//! Main file, yeah
//!
//! Yes, it's full of magick numbers. Hope you enjoy them! ;)
//!
//! Let's just pray that it all works.

#![deny(
	unreachable_patterns, // because if it is, it's probably a bug that appeared after changing something
)]

use core::panic;

use chrono::{DateTime, Local};
use clap::{Parser, ValueEnum};
use image::ImageBuffer;
use rand::{rngs::ThreadRng, seq::SliceRandom, thread_rng, Rng};
use rand_distr::{num_traits::Float, Normal, Poisson};
use rayon::iter::ParallelIterator;

mod angle;
mod float_type;
mod typesafe_rng;

use angle::{AngleDeg, AngleRad};
use float_type::float;
use typesafe_rng::TypesafeRNG;


#[derive(Parser, Debug)]
#[clap(
	about,
	author,
	version,
	help_template = "\
		{before-help}{name} v{version}\n\
		{about}\n\
		Author: {author}\n\
		\n\
		{usage-heading} {usage}\n\
		\n\
		{all-args}{after-help}\
	",
)]
struct CliArgsPre {
	#[arg(short='c', long, default_value_t=false)]
	list_all_named_colors: bool,

	#[arg(short='s', long, default_value="1920,1080")]
	image_sizes: String,

	/// Background average color.
	///
	/// Possible options for all color related options:
	/// - Rgb: "rgb <r> <g> <b>", where <r>, <g> and <b> - integers (0..=255) or floating point numbers (0. ..= 1.).
	/// - Hsl: "hsl <h> <s> <l>", where
	///   - <h> - float (0. ..= 360.)
	///   - <s> - float (0. ..= 1.) or percentage (0.% ..= 100.%)
	///   - <l> - float (0. ..= 1.) or percentage (0.% ..= 100.%)
	/// - and all named colors - run with `--list-all-named-colors` to get full list.
	#[arg(short='b', long, default_value="rgb 32 27 30")]
	background_color: String,
	#[arg(short='d', long, default_value="blue,red")]
	dominant_colors: String,
	#[arg(short='m', long, default_value="yellow,green")]
	minor_colors: String,

	/// Angle of rays, in degrees.
	#[arg(short='a', long, allow_hyphen_values=true)]
	angle_of_rays: Option<float>,

	#[arg(long, default_value_t=StaticNoiseType::Mul)]
	pre_noise_type: StaticNoiseType,
	#[arg(long, default_value_t=1.1)]
	pre_noise_value: float,

	#[arg(long, default_value_t=StaticNoiseType::Mul)]
	mid_noise_type: StaticNoiseType,
	#[arg(long, default_value_t=1.1)]
	mid_noise_value: float,
	#[arg(long, default_value_t=0.1)]
	mid_noise_probability: float,

	#[arg(long, default_value_t=StaticNoiseType::Mul)]
	post_noise_type: StaticNoiseType,
	#[arg(long, default_value_t=1.1)]
	post_noise_value: float,
}

/// Post-Processed CliArgs
struct CliArgsPost {
	image_w: u32,
	image_h: u32,

	background_color: Color,
	dominant_colors: Vec<Color>,
	minor_colors: Vec<Color>,

	angle_of_rays: Option<AngleDeg>,

	pre_noise: Option<StaticNoise>,
	mid_noise: Option<(StaticNoise, float)>,
	post_noise: Option<StaticNoise>,
}
impl From<CliArgsPre> for CliArgsPost {
	fn from(CliArgsPre {
		list_all_named_colors: _,

		image_sizes,

		background_color: bg_color,
		dominant_colors,
		minor_colors,

		angle_of_rays,

		pre_noise_type,
		pre_noise_value,

		mid_noise_type,
		mid_noise_value,
		mid_noise_probability,

		post_noise_type,
		post_noise_value,
	}: CliArgsPre) -> Self {
		Self {
			image_w: image_sizes.split_once(',').expect("cant split image sizes by comma").0.parse().expect("cant parse image width as integer"),
			image_h: image_sizes.split_once(',').expect("cant split image sizes by comma").1.parse().expect("cant parse image height as integer"),

			background_color: Color::from_str(&bg_color),
			dominant_colors: dominant_colors.split(',').map(|s| Color::from_str(s)).collect(),
			minor_colors: minor_colors.split(',').map(|s| Color::from_str(s)).collect(),

			angle_of_rays: angle_of_rays.map(AngleDeg::new),

			pre_noise: StaticNoise::from_type_and_value(pre_noise_type, pre_noise_value),
			mid_noise: StaticNoise::from_type_and_value_and_prob(mid_noise_type, mid_noise_value, mid_noise_probability),
			post_noise: StaticNoise::from_type_and_value(post_noise_type, post_noise_value),
		}
	}
}


type ImageBuf = ImageBuffer::<image::Rgb<u8>, Vec<u8>>;


fn main() {
	let timestamp_program_start = Local::now();
	let cli_args = CliArgsPre::parse();
	if cli_args.list_all_named_colors {
		list_all_named_colors();
		return;
	}
	let cli_args = CliArgsPost::from(cli_args);

	let mut img = ImageBuf::new(cli_args.image_w, cli_args.image_h);

	img.draw(cli_args);

	let filename = format!("slb_{now}.png", now=timestamp_program_start.to_my_format());
	img.save(&filename).expect("unable to save image");
	println!("DONE: {filename}");
}





trait ImgMethods {
	fn draw(&mut self, cli_args: CliArgsPost);
	fn fill_bg(&mut self, bg_color: Color, image_wh_f: (float, float));
	fn maybe_apply_static_noise(&mut self, noise: Option<StaticNoise>);
	fn apply_random_nonstatic_noise(&mut self, image_wh_f: (float, float));

	fn fill_by_color(&mut self, color: Color);
	fn fill_by_noise_grey(&mut self, noise: impl Noise+Sync);
	fn fill_by_noise_rgb(&mut self, noise_r: impl Noise+Sync, noise_g: impl Noise+Sync, noise_b: impl Noise+Sync);
}
impl ImgMethods for ImageBuf {
	fn draw(&mut self, cli_args: CliArgsPost) {
		let image_wh_i = (self.width(), self.height());
		let image_wh_f = (self.width() as float, self.height() as float);

		// TODO: seeded rng EVERYWHERE
		let mut rng = thread_rng();

		macro_rules! apply_random_nonstatic_noise {
		    () => {
				for _ in 0..1+rng.sample(Poisson::new(0.5).unwrap()).round() as u32 {
					self.apply_random_nonstatic_noise(image_wh_f);
				}
		    };
		}
		macro_rules! maybe_apply_static_mid_noise {
		    () => {
				if let Some((mid_noise, mid_noise_probability)) = cli_args.mid_noise {
					if rng.gen_bool(mid_noise_probability as f64) {
						self.maybe_apply_static_noise(Some(mid_noise));
					}
				}
		    };
		}

		self.fill_bg(cli_args.background_color, image_wh_f);
		self.maybe_apply_static_noise(cli_args.pre_noise);
		apply_random_nonstatic_noise!();

		for _ in 0..4+rng.sample(Poisson::new(0.5).unwrap()).round() as u32 {
			maybe_apply_static_mid_noise!();
			macro_rules! gen_random_perimeter_delta { () => { rng.sample(Normal::new(20., 10.).unwrap()).abs() }; }
			use typesafe_rng::V3::*;
			match rng.gen_typesafe_with_weights([2, 1, 1]) {
				_1 => {
					self.apply_with_force(PullInWithAngleAndForce {
						pull_from: PullIn::new_random(image_wh_f),
						angle: cli_args.angle_of_rays.unwrap_or_else(|| AngleDeg::new_random()),
						pull_force: rng.gen_range(0.95 ..= 1.),
						pixels_iter_order: PixelsIterOrder::new_random(),
					}, 1.0);
				}
				_2 => {
					let shapes_n: usize = rng.sample(Poisson::new(3_f32).unwrap()).round() as usize;
					let mut shapes = Vec::<ShapeWithColorAndMaybePerimeter>::with_capacity(shapes_n);
					for _ in 0..shapes_n {
						shapes.push(ShapeWithColorAndMaybePerimeter {
							shape: Shape::new_random(image_wh_f),
							color: Color::new_random(),
							is_perimeter: rng.gen_bool(0.5).then_some(gen_random_perimeter_delta!()),
						});
					}
					for shape in shapes {
						self.apply(shape);
					}
				}
				_3 => {
					let shapes_n: usize = rng.sample(Poisson::new(3_f32).unwrap()).round() as usize;
					let is_perimeter = rng.gen_bool(0.5).then_some(gen_random_perimeter_delta!());
					let mut shapes = Vec::<ShapeWithColorAndMaybePerimeter>::with_capacity(shapes_n);
					for _ in 0..shapes_n {
						shapes.push(ShapeWithColorAndMaybePerimeter {
							shape: Shape::new_random(image_wh_f),
							color: Color::new_random(),
							is_perimeter,
						});
					}
					for shape in shapes {
						self.apply(shape);
					}
				}
			}
		}

		maybe_apply_static_mid_noise!();
		apply_random_nonstatic_noise!();
		self.maybe_apply_static_noise(cli_args.post_noise);
	}
	fn fill_bg(&mut self, bg_color: Color, image_wh_f: (float, float)) {
		use typesafe_rng::V5::*;
		match thread_rng().gen_typesafe_number() {
			_1 => self.fill_by_color(bg_color),
			_2 => self.fill_by_noise_grey(SinNoise::new_random_default()),
			_3 => self.fill_by_noise_grey(RandomPointsNoise::new_random_default(image_wh_f)),
			_4 => self.fill_by_noise_rgb(
				SinNoise::new_random_default(),
				SinNoise::new_random_default(),
				SinNoise::new_random_default(),
			),
			_5 => self.fill_by_noise_rgb(
				RandomPointsNoise::new_random_default(image_wh_f),
				RandomPointsNoise::new_random_default(image_wh_f),
				RandomPointsNoise::new_random_default(image_wh_f),
			),
		}
	}
	fn maybe_apply_static_noise(&mut self, noise: Option<StaticNoise>) {
		if let Some(noise) = noise {
			self.apply(noise);
		}
	}
	fn apply_random_nonstatic_noise(&mut self, image_wh_f: (float, float)) {
		let mut rng = thread_rng();
		let new_random_points_noise = ||  RandomPointsNoise::new_random_default(image_wh_f);
		let new_sin_noise = || SinNoise::new_random_default();
		use typesafe_rng::V4::*;
		let force = rng.gen_range(0. .. 1.);
		match rng.gen_typesafe_number() {
			_1 => self.apply_with_force(new_random_points_noise(), force),
			_2 => self.apply_with_force(new_sin_noise(), force),
			_3 => self.apply_with_force((
				new_random_points_noise(),
				new_random_points_noise(),
				new_random_points_noise(),
			), force),
			_4 => self.apply_with_force((
				new_sin_noise(),
				new_sin_noise(),
				new_sin_noise(),
			), force),
			// _5 => {
			// 	let noise_r: Box<dyn Noise> = if rng.gen_bool(0.5) { Box::new(new_sin_noise()) } else { Box::new(new_random_points_noise()) };
			// 	let noise_g: Box<dyn Noise> = if rng.gen_bool(0.5) { Box::new(new_sin_noise()) } else { Box::new(new_random_points_noise()) };
			// 	let noise_b: Box<dyn Noise> = if rng.gen_bool(0.5) { Box::new(new_sin_noise()) } else { Box::new(new_random_points_noise()) };
			// 	self.apply_with_force((
			// 		*noise_r,
			// 		*noise_g,
			// 		*noise_b,
			// 	), force)
			// }
		}
	}

	fn fill_by_color(&mut self, color: Color) {
		let color = color.to_image_rgb();
		self.par_pixels_mut().for_each(|pixel| {
			*pixel = color;
		});
	}
	fn fill_by_noise_grey(&mut self, noise: impl Noise+Sync) {
		let w_half = self.width()  as float / 2.;
		let h_half = self.height() as float / 2.;
		self.par_enumerate_pixels_mut().for_each(|(x, y, pixel)| {
			let x = x as float;
			let y = y as float;
			*pixel = Color::grey(
				noise.at(x-w_half, y-h_half)
			).to_image_rgb();
		});
	}
	fn fill_by_noise_rgb(&mut self, noise_r: impl Noise+Sync, noise_g: impl Noise+Sync, noise_b: impl Noise+Sync) {
		let w_half = self.width()  as float / 2.;
		let h_half = self.height() as float / 2.;
		self.par_enumerate_pixels_mut().for_each(|(x, y, pixel)| {
			let x = x as float;
			let y = y as float;
			*pixel = Color::rgb_f(
				noise_r.at(x-w_half, y-h_half),
				noise_g.at(x-w_half, y-h_half),
				noise_b.at(x-w_half, y-h_half),
			).to_image_rgb();
		});
	}
}


trait Apply<T> {
	fn apply_with_force(&mut self, t: T, force: float);
	fn apply(&mut self, t: T) {
		self.apply_with_force(t, 1.);
	}
}


impl Apply<StaticNoise> for ImageBuf {
	fn apply_with_force(&mut self, noise: StaticNoise, force: float) {
		self.par_pixels_mut().for_each(|pixel| {
			pixel.apply_with_force(noise, force);
		});
	}
}

trait Boundary {
	fn contains(&self, px: float, py: float) -> bool;
	fn is_on_perimeter(&self, px: float, py: float, delta: float, image_wh: (float, float)) -> bool;
}

pub enum Shape {
	WholeImage,
	Rectangle { x: float, y: float, w: float, h: float },
}
impl Shape {
	pub fn rect_whole_image((image_w, image_h): (float, float)) -> Self {
		Self::Rectangle { x: 0., y: 0., w: image_w, h: image_h }
	}
	pub fn new_random(image_wh_f: (float, float)) -> Self {
		let mut rng = thread_rng();
		let (image_w, image_h) = image_wh_f;
		let image_w_half = image_w / 2.;
		let image_h_half = image_h / 2.;
		use typesafe_rng::V1::*;
		match rng.gen_typesafe_number() {
			_1 => Self::Rectangle {
				x: rng.gen_range(-image_w_half ..= image_w_half),
				y: rng.gen_range(-image_h_half ..= image_h_half),
				w: rng.sample(Normal::new(image_w_half/2., image_w_half/2.).unwrap()).abs(),
				h: rng.sample(Normal::new(image_h_half/2., image_h_half/2.).unwrap()).abs(),
			}
		}
	}
}
impl Boundary for Shape {
	fn contains(&self, px: float, py: float) -> bool {
		match self {
			Self::WholeImage => true,
			Self::Rectangle { x, y, w, h } => {
				x-w/2. <= px && px <= x+w/2. &&
				y-h/2. <= py && py <= y+h/2.
			}
		}
	}
	fn is_on_perimeter(&self, px: float, py: float, delta: float, image_wh: (float, float)) -> bool {
		match self {
			Self::WholeImage => Self::rect_whole_image(image_wh).is_on_perimeter(px, py, delta, image_wh),
			Self::Rectangle { x, y, w, h } => {
				abs(x-w/2.-px) <= delta || abs(x+w/2.-px) <= delta ||
				abs(y-h/2.-py) <= delta || abs(y+h/2.-py) <= delta
			}
		}
	}
}

pub struct ShapeWithColorAndMaybePerimeter {
	shape: Shape,
	color: Color,
	is_perimeter: Option<float>,
}

impl Apply<ShapeWithColorAndMaybePerimeter> for ImageBuf {
	fn apply_with_force(
		&mut self,
		ShapeWithColorAndMaybePerimeter {
			shape,
			color,
			is_perimeter,
		}: ShapeWithColorAndMaybePerimeter,
		force: float,
	) {
		let color = color.to_image_rgb();
		let image_wh_f = (self.width() as float, self.height() as float);
		let w_half = self.width()  as float / 2.;
		let h_half = self.height() as float / 2.;
		// TODO(optimization): loop only from min to max
		for (px, py, pixel) in self.enumerate_pixels_mut() {
			if let Some(perimeter) = is_perimeter {
				if shape.is_on_perimeter(px as float - w_half, py as float - h_half, perimeter, image_wh_f) {
					*pixel = color.lerp_inv(*pixel, force);
				}
			} else {
				if shape.contains(px as float - w_half, py as float - h_half) {
					*pixel = color.lerp_inv(*pixel, force);
				}
			}
		}
	}
}


trait Lerp {
	fn lerp(self, other: Self, t: float) -> Self;
	fn lerp_inv(self, other: Self, t: float) -> Self where Self: Sized {
		self.lerp(other, 1.-t)
	}
}
// impl<T: Mul<float, Output=T> + Add<T, Output=T>> Lerp for T {
// 	fn lerp(self, other: Self, t: float) -> Self {
// 		self*(1.-t) + other*t
// 	}
// }
impl Lerp for float {
	fn lerp(self, other: Self, t: float) -> Self {
		self*(1.-t) + other*t
	}
}
impl Lerp for u8 {
	fn lerp(self, other: Self, t: float) -> Self {
		(self as float).lerp(other as float, t).round() as u8
	}
}
impl Lerp for image::Rgb<u8> {
	fn lerp(self, other: Self, t: float) -> Self {
		let image::Rgb([r, g, b]) = self;
		let image::Rgb([or, og, ob]) = other;
		Self([
			r.lerp(or, t),
			g.lerp(og, t),
			b.lerp(ob, t),
		])
	}
}


#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelsIterOrder {
	Random { times: u32 },
	Default, Default2,
	X_Y, Xinv_Y, X_Yinv, Xinv_Yinv,
	Y_X, Yinv_X, Y_Xinv, Yinv_Xinv,
}
impl PixelsIterOrder {
	pub fn new_random() -> Self {
		use typesafe_rng::V11::*;
		use PixelsIterOrder::*;
		match thread_rng().gen_typesafe_with_weights([1, 10,5, 1,1,1,1, 1,1,1,1]) {
			_1 => Random { times: 1+thread_rng().sample(Poisson::new(7.).unwrap()).round() as u32 },
			_2 => Default, _3 => Default2,
			_4 => X_Y, _5 => Xinv_Y, _6 => X_Yinv, _7  => Xinv_Yinv,
			_8 => Y_X, _9 => Yinv_X, _10 => Y_Xinv, _11 => Yinv_Xinv,
		}
	}
	pub fn is_default(self) -> bool {
		self == Self::Default
	}
	pub fn is_default2(self) -> bool {
		self == Self::Default2
	}
	pub fn convert_if_default(&mut self, default: PixelsIterOrder, default2: PixelsIterOrder) {
		assert_ne!(default, PixelsIterOrder::Default);
		assert_ne!(default, PixelsIterOrder::Default2);
		// i know, geniusly, but thats the way life is
		assert_ne!(default2, PixelsIterOrder::Default);
		assert_ne!(default2, PixelsIterOrder::Default2);
		if self.is_default() {
			*self = default;
		} else if self.is_default2() {
			*self = default2;
		}
	}
	pub fn generate_queue(self, w: u32, h: u32, rng: &mut ThreadRng) -> Vec<(u32, u32)> {
		use PixelsIterOrder::*;
		let mut queue = Vec::<(u32, u32)>::with_capacity(w as usize * h as usize);
		match self {
			Default | Default2 => panic!(),
			Random { times } => {
				let mut queue_0 = queue;
				for px in 0..w {
					for py in 0..h {
						queue_0.push((px, py));
					}
				}
				queue = Vec::<(u32, u32)>::with_capacity(times as usize * w as usize * h as usize);
				for _ in 0..times {
					queue.extend(&queue_0);
				}
				queue.shuffle(rng);
			}
			X_Y => {
				for px in 0..w {
					for py in 0..h {
						queue.push((px, py));
					}
				}
			}
			Xinv_Y => {
				for px in (0..w).rev() {
					for py in 0..h {
						queue.push((px, py));
					}
				}
			}
			X_Yinv => {
				for px in 0..w {
					for py in (0..h).rev() {
						queue.push((px, py));
					}
				}
			}
			Xinv_Yinv => {
				for px in (0..w).rev() {
					for py in (0..h).rev() {
						queue.push((px, py));
					}
				}
			}
			Y_X => {
				for py in 0..h {
					for px in 0..w {
						queue.push((px, py));
					}
				}
			}
			Yinv_X => {
				for py in (0..h).rev() {
					for px in 0..w {
						queue.push((px, py));
					}
				}
			}
			Y_Xinv => {
				for py in 0..h {
					for px in (0..w).rev() {
						queue.push((px, py));
					}
				}
			}
			Yinv_Xinv => {
				for py in (0..h).rev() {
					for px in (0..w).rev() {
						queue.push((px, py));
					}
				}
			}
		}
		queue
	}
}

pub enum PullIn {
	AllArea,
	Shape(Shape),
	ShapePerimeter { shape: Shape, delta: float },
}
impl PullIn {
	pub fn new_random(image_wh_f: (float, float)) -> Self {
		use typesafe_rng::V3::*;
		let mut rng = thread_rng();
		match rng.gen_typesafe_with_weights([3,1,1]) {
			_1 => Self::AllArea,
			_2 => Self::Shape(Shape::new_random(image_wh_f)),
			_3 => Self::ShapePerimeter {
				shape: Shape::new_random(image_wh_f),
				delta: rng.sample(Normal::new(20., 10.).unwrap()).abs(),
			},
		}
	}
}
pub struct PullInWithAngleAndForce {
	pull_from: PullIn,
	angle: AngleDeg,
	pull_force: float,
	pixels_iter_order: PixelsIterOrder,
}
impl Apply<PullInWithAngleAndForce> for ImageBuf {
	fn apply_with_force(
		&mut self,
		PullInWithAngleAndForce {
			pull_from,
			angle,
			pull_force, // TODO: #dcf12b
			mut pixels_iter_order,
		}: PullInWithAngleAndForce,
		force: float, // TODO: #dcf12b
	) {
		// let w_half_f = self.width()  as float / 2.;
		// let h_half_f = self.height() as float / 2.;
		use PullIn::*;
		use angle::Octant::*;
		use PixelsIterOrder::*;
		let (w, h) = (self.width(), self.height());
		let w_half_f = w as float / 2.;
		let h_half_f = h as float / 2.;
		let image_wh_f = (w as float, h as float);
		let angle_cos2 = angle.cos().powi(2) as f64;
		let angle_sin2 = angle.sin().powi(2) as f64;
		let mut rng = thread_rng();
		match pull_from {
			AllArea => {
				match angle.octant() {
					_1 => {
						pixels_iter_order.convert_if_default(Yinv_X, X_Yinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
								self[( x.saturating_sub(1), y )]
							} else {
								self[( x, (y+1).min(h-1) )]
							}, pull_force);
						}
					}
					_2 => {
						pixels_iter_order.convert_if_default(X_Yinv, Yinv_X);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
								self[( x, (y+1).min(h-1) )]
							} else {
								self[( x.saturating_sub(1), y )]
							}, pull_force);
						}
					}
					_3 => {
						pixels_iter_order.convert_if_default(Xinv_Yinv, Yinv_Xinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
								self[( x, (y+1).min(h-1) )]
							} else {
								self[( (x+1).min(w-1), y )]
							}, pull_force);
						}
					}
					_4 => {
						pixels_iter_order.convert_if_default(Yinv_Xinv, Xinv_Yinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
								self[( (x+1).min(w-1), y )]
							} else {
								self[( x, (y+1).min(h-1) )]
							}, pull_force);
						}
					}
					_5 => {
						pixels_iter_order.convert_if_default(Y_Xinv, Xinv_Y);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
								self[( (x+1).min(w-1), y )]
							} else {
								self[( x, y.saturating_sub(1) )]
							}, pull_force);
						}
					}
					_6 => {
						pixels_iter_order.convert_if_default(Xinv_Y, Y_Xinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
								self[( x, y.saturating_sub(1) )]
							} else {
								self[( (x+1).min(w-1), y )]
							}, pull_force);
						}
					}
					_7 => {
						pixels_iter_order.convert_if_default(X_Y, Y_X);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
								self[( x, y.saturating_sub(1) )]
							} else {
								self[( x.saturating_sub(1), y )]
							}, pull_force);
						}
					}
					_8 => {
						pixels_iter_order.convert_if_default(Y_X, X_Y);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
								self[( x.saturating_sub(1), y )]
							} else {
								self[( x, y.saturating_sub(1) )]
							}, pull_force);
						}
					}
				}
			}
			Shape(shape) => {
				match angle.octant() {
					_1 => {
						pixels_iter_order.convert_if_default(Yinv_X, X_Yinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( x.saturating_sub(1), y )]
								} else {
									self[( x, (y+1).min(h-1) )]
								}, pull_force);
							}
						}
					}
					_2 => {
						pixels_iter_order.convert_if_default(X_Yinv, Yinv_X);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, (y+1).min(h-1) )]
								} else {
									self[( x.saturating_sub(1), y )]
								}, pull_force);
							}
						}
					}
					_3 => {
						pixels_iter_order.convert_if_default(Xinv_Yinv, Yinv_Xinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, (y+1).min(h-1) )]
								} else {
									self[( (x+1).min(w-1), y )]
								}, pull_force);
							}
						}
					}
					_4 => {
						pixels_iter_order.convert_if_default(Yinv_Xinv, Xinv_Yinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( (x+1).min(w-1), y )]
								} else {
									self[( x, (y+1).min(h-1) )]
								}, pull_force);
							}
						}
					}
					_5 => {
						pixels_iter_order.convert_if_default(Y_Xinv, Xinv_Y);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( (x+1).min(w-1), y )]
								} else {
									self[( x, y.saturating_sub(1) )]
								}, pull_force);
							}
						}
					}
					_6 => {
						pixels_iter_order.convert_if_default(Xinv_Y, Y_Xinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, y.saturating_sub(1) )]
								} else {
									self[( (x+1).min(w-1), y )]
								}, pull_force);
							}
						}
					}
					_7 => {
						pixels_iter_order.convert_if_default(X_Y, Y_X);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, y.saturating_sub(1) )]
								} else {
									self[( x.saturating_sub(1), y )]
								}, pull_force);
							}
						}
					}
					_8 => {
						pixels_iter_order.convert_if_default(Y_X, X_Y);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.contains(px, py) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( x.saturating_sub(1), y )]
								} else {
									self[( x, y.saturating_sub(1) )]
								}, pull_force);
							}
						}
					}
				}
			}
			ShapePerimeter { shape, delta } => {
				match angle.octant() {
					_1 => {
						pixels_iter_order.convert_if_default(Yinv_X, X_Yinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( x.saturating_sub(1), y )]
								} else {
									self[( x, (y+1).min(h-1) )]
								}, pull_force);
							}
						}
					}
					_2 => {
						pixels_iter_order.convert_if_default(X_Yinv, Yinv_X);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, (y+1).min(h-1) )]
								} else {
									self[( x.saturating_sub(1), y )]
								}, pull_force);
							}
						}
					}
					_3 => {
						pixels_iter_order.convert_if_default(Xinv_Yinv, Yinv_Xinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, (y+1).min(h-1) )]
								} else {
									self[( (x+1).min(w-1), y )]
								}, pull_force);
							}
						}
					}
					_4 => {
						pixels_iter_order.convert_if_default(Yinv_Xinv, Xinv_Yinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( (x+1).min(w-1), y )]
								} else {
									self[( x, (y+1).min(h-1) )]
								}, pull_force);
							}
						}
					}
					_5 => {
						pixels_iter_order.convert_if_default(Y_Xinv, Xinv_Y);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( (x+1).min(w-1), y )]
								} else {
									self[( x, y.saturating_sub(1) )]
								}, pull_force);
							}
						}
					}
					_6 => {
						pixels_iter_order.convert_if_default(Xinv_Y, Y_Xinv);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, y.saturating_sub(1) )]
								} else {
									self[( (x+1).min(w-1), y )]
								}, pull_force);
							}
						}
					}
					_7 => {
						pixels_iter_order.convert_if_default(X_Y, Y_X);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_sin2) {
									self[( x, y.saturating_sub(1) )]
								} else {
									self[( x.saturating_sub(1), y )]
								}, pull_force);
							}
						}
					}
					_8 => {
						pixels_iter_order.convert_if_default(Y_X, X_Y);
						let queue = pixels_iter_order.generate_queue(w, h, &mut rng);
						for (x, y) in queue.into_iter() {
							let px = x as float - w_half_f;
							let py = y as float - h_half_f;
							if shape.is_on_perimeter(px, py, delta, image_wh_f) {
								self[(x, y)] = self[(x, y)].lerp(if rng.gen_bool(angle_cos2) {
									self[( x.saturating_sub(1), y )]
								} else {
									self[( x, y.saturating_sub(1) )]
								}, pull_force);
							}
						}
					}
				}
			}
		}
	}
}


pub trait Noise {
	/// Returns random float number from 0 to 1, that semi continuously depends on `x` and `y`.
	///
	/// At least, let's hope that it satisfies abovementioned conditions.
	fn at(&self, x: float, y: float) -> float;
}


impl<T: Noise+Sync+Clone> Apply<T> for ImageBuf {
	fn apply_with_force(&mut self, noise: T, force: float) {
		self.apply_with_force((Shape::WholeImage, noise.clone(), noise.clone(), noise), force);
	}
}


impl<T1, T2, T3> Apply<(T1, T2, T3)> for ImageBuf
where T1: Noise+Sync, T2: Noise+Sync, T3: Noise+Sync
{
	fn apply_with_force(&mut self, (noise_r, noise_g, noise_b): (T1, T2, T3), force: float) {
		self.apply_with_force((Shape::WholeImage, noise_r, noise_g, noise_b), force)
	}
}


impl<B, T1, T2, T3> Apply<(B, T1, T2, T3)> for ImageBuf
where B: Boundary+Sync, T1: Noise+Sync, T2: Noise+Sync, T3: Noise+Sync
{
	fn apply_with_force(&mut self, (boundary, noise_r, noise_g, noise_b): (B, T1, T2, T3), force: float) {
		let image_w_half = self.width()  as float / 2.;
		let image_h_half = self.height() as float / 2.;
		self.par_enumerate_pixels_mut().for_each(|(px, py, pixel)| {
			let x = px as float - image_w_half;
			let y = py as float - image_h_half;
			if boundary.contains(x, y) {
				let noise_here_r = noise_r.at(x, y);
				let noise_here_g = noise_g.at(x, y);
				let noise_here_b = noise_b.at(x, y);
				assert!(0. <= noise_here_r && noise_here_r <= 1.);
				assert!(0. <= noise_here_g && noise_here_g <= 1.);
				assert!(0. <= noise_here_b && noise_here_b <= 1.);
				pixel.0[0] = ((pixel.0[0] as float * 2. * noise_here_r).clamp(0., 255.) as u8).lerp_inv(pixel.0[0], force);
				pixel.0[1] = ((pixel.0[1] as float * 2. * noise_here_g).clamp(0., 255.) as u8).lerp_inv(pixel.0[1], force);
				pixel.0[2] = ((pixel.0[2] as float * 2. * noise_here_b).clamp(0., 255.) as u8).lerp_inv(pixel.0[2], force);
			}
		});
	}
}

#[derive(Debug, Clone)]
pub struct SinNoise {
	frequencies_x: Vec<float>,
	frequencies_y: Vec<float>,
}
impl SinNoise {
	pub fn new(frequencies_x: Vec<float>, frequencies_y: Vec<float>) -> Self {
		Self { frequencies_x, frequencies_y }
	}
	pub fn new_random_default() -> Self {
		Self::new_random(10., 3e-2, 5e-3, 1.5, 0.5)
	}
	pub fn new_random(
		freq_n_poisson_lambda: float,
		freq_0_mean: float,
		freq_0_dev: float,
		freq_decay_mean: float,
		freq_decay_dev: float,
	) -> Self {
		fn mean_and_dev_to_min_and_max(mean: float, dev: float) -> (float, float) {
			let mut rng = thread_rng();
			let delta = rng.sample(Normal::new(0., dev).unwrap());
			let v1 = mean - delta;
			let v2 = mean + delta;
			(v1.min(v2), v1.max(v2))
		}
		let (f0_min, f0_max) = mean_and_dev_to_min_and_max(freq_0_mean, freq_0_dev);
		let (d_min, d_max) = mean_and_dev_to_min_and_max(freq_decay_mean, freq_decay_dev);
		let mut rng = thread_rng();
		let mut fx = vec![rng.gen_range(f0_min ..= f0_max)];
		for _ in 0..rng.sample(Poisson::new(freq_n_poisson_lambda).unwrap()).round() as u32 {
			fx.push(fx.last().unwrap() / rng.gen_range(d_min ..= d_max));
		}
		let mut fy = vec![rng.gen_range(f0_min ..= f0_max)];
		for _ in 0..rng.sample(Poisson::new(freq_n_poisson_lambda).unwrap()).round() as u32 {
			fy.push(fy.last().unwrap() / rng.gen_range(d_min ..= d_max));
		}
		Self { frequencies_x: fx, frequencies_y: fy }
	}
	pub fn sum_at(&self, x: float, y: float) -> float {
		let value_from_x: float = self.frequencies_x.iter().map(|f| (sin_raw(f*x-f)+1.)/2.).sum();
		let value_from_y: float = self.frequencies_y.iter().map(|f| (sin_raw(f*y-f)+1.)/2.).sum();
		(value_from_x + value_from_y) / (self.frequencies_x.len() + self.frequencies_y.len()) as float
	}
	pub fn prod_at(&self, x: float, y: float) -> float {
		let value_from_x: float = self.frequencies_x.iter().map(|f| (sin_raw(f*x-f)+1.)/2.).product();
		let value_from_y: float = self.frequencies_y.iter().map(|f| (sin_raw(f*y-f)+1.)/2.).product();
		value_from_x * value_from_y
	}
	pub fn prod_sqrt_at(&self, x: float, y: float) -> float {
		self.prod_at(x, y).powf(1. / (self.frequencies_x.len() + self.frequencies_y.len()) as float)
	}
}
impl Noise for SinNoise {
	fn at(&self, x: float, y: float) -> float {
		// self.sum_at(x, y)
		// self.prod_at(x, y)
		// self.prod_sqrt_at(x, y)
		(self.sum_at(x, y) + self.prod_at(x, y)) / 2.
	}
}


#[derive(Debug, Clone)]
pub struct RandomPointsNoise {
	metric: Metric,
	points: Vec<(float, float)>,
	// precalculated
	avg_dist: float,
}
impl RandomPointsNoise {
	pub fn new(metric: Metric, points: Vec<(float, float)>) -> Self {
		let mut total_dist: float = 0.;
		let mut total_n: u32 = 0;
		if points.len() < 100 {
			for i in 0..points.len() {
				for j in i+1..points.len() {
					let a = points[i];
					let b = points[j];
					let dist = metric.dist(a, b);
					total_dist += dist;
					total_n += 1;
				}
			}
		} else {
			let mut rng = thread_rng();
			for _ in 0..10_000 {
				let i = rng.gen_range(0 .. points.len());
				let j = rng.gen_range(0 .. points.len());
				if i == j { continue }
				let a = points[i];
				let b = points[j];
				let dist = metric.dist(a, b);
				total_dist += dist;
				total_n += 1;
			}
		}
		let avg_dist: float = total_dist / total_n as float;
		Self { metric, points, avg_dist }
	}
	pub fn new_random_default(image_wh: (float, float)) -> Self {
		Self::new_random(image_wh, 20.)
	}
	pub fn new_random(
		image_wh: (float, float),
		n_poisson_lambda: float,
	) -> Self {
		let image_w_half = image_wh.0 / 2.;
		let image_h_half = image_wh.1 / 2.;
		let mut rng = thread_rng();
		let points_n: usize = rng.sample(Poisson::new(n_poisson_lambda).unwrap()).round() as usize;
		let mut points = Vec::<(float, float)>::with_capacity(points_n);
		for _ in 0..points_n {
			points.push((
				rng.gen_range(-image_w_half ..= image_w_half),
				rng.gen_range(-image_h_half ..= image_h_half),
			));
		}
		Self::new(Metric::new_random(), points)
	}
}
impl Noise for RandomPointsNoise {
	fn at(&self, x: float, y: float) -> float {
		let dist_to_closest_point: float = self.points.iter()
			.map(|&(px, py)| self.metric.dist((px, py), (x, y)))
			.min_by(|d1, d2| d1.total_cmp(d2))
			.unwrap();
		(dist_to_closest_point / self.avg_dist).clamp(0., 1.)
	}
}


#[derive(Debug, Clone)]
pub enum Metric {
	Abs,
	Euclid,
	Pow(float),
	GeometricMean,
}
impl Metric {
	pub fn new_random() -> Self {
		use typesafe_rng::V4::*;
		let mut rng = thread_rng();
		match rng.gen_typesafe_with_weights([5, 10, 1, 1]) {
			_1 => Self::Abs,
			_2 => Self::Euclid,
			_3 => Self::Pow(3.+rng.sample(Poisson::new(0.7).unwrap())),
			_4 => Self::GeometricMean,
		}
	}
	pub fn dist(&self, (ax, ay): (float, float), (bx, by): (float, float)) -> float {
		match self {
			Self::Abs => abs(ax-bx) + abs(ay-by),
			Self::Euclid => sqrt( (ax-bx).powi(2) + (ay-by).powi(2) ),
			Self::Pow(n) => ( (ax-bx).abs().powf(*n) + (ay-by).abs().powf(*n) ).powf(1. / n),
			Self::GeometricMean => sqrt( (ax-bx).abs() * (ay-by).abs() ),
		}
	}
}


#[derive(Debug, Clone, Copy, ValueEnum)]
enum StaticNoiseType {
	None,
	Add,
	Mul,
}
impl ToString for StaticNoiseType {
	fn to_string(&self) -> String {
		match self {
			Self::None => format!("none"),
			Self::Add  => format!("add"),
			Self::Mul  => format!("mul"),
		}
	}
}

#[derive(Debug, Clone, Copy)]
enum StaticNoise {
	Add { value: float },
	Mul { value: float },
}
impl StaticNoise {
	fn from_type_and_value(type_: StaticNoiseType, value: float) -> Option<Self> {
		match type_ {
			StaticNoiseType::None => None,
			StaticNoiseType::Add => Some(Self::Add { value }),
			StaticNoiseType::Mul => Some(Self::Mul { value }),
		}
	}
	fn from_type_and_value_and_prob(type_: StaticNoiseType, value: float, prob: float) -> Option<(Self, float)> {
		Self::from_type_and_value(type_, value)
			.map(|n| (n, prob))
	}
}
impl Apply<StaticNoise> for image::Rgb<u8> {
	fn apply_with_force(&mut self, noise: StaticNoise, force: float) {
	    self.0[0].apply_with_force(noise, force);
	    self.0[1].apply_with_force(noise, force);
	    self.0[2].apply_with_force(noise, force);
	}
}
impl Apply<StaticNoise> for u8 {
	fn apply_with_force(&mut self, noise: StaticNoise, force: float) {
		*self = match noise {
			StaticNoise::Add { value } => {
				let random_value: float = thread_rng().gen_range(-value ..= value);
				((*self as float / 255. + random_value) * 255.).round().clamp(0., 255.) as u8
			}
			StaticNoise::Mul { value } => {
				let v1 = value;
				let v2 = 1. / value;
				let v_min = v1.min(v2);
				let v_max = v1.max(v2);
				let random_value: float = thread_rng().gen_range(v_min ..= v_max);
				(*self as float * random_value).round().clamp(0., 255.) as u8
			}
		}.lerp_inv(*self, force)
	}
}



#[derive(Debug, Clone, Copy)]
pub enum Color {
	Hsl(Hsl),
	Rgb(Rgb),
}
impl Color {
	pub const fn rgb_f(r: float, g: float, b: float) -> Self {
		Self::Rgb(Rgb { r, g, b })
	}
	pub const fn rgb_i(r: u8, g: u8, b: u8) -> Self {
		Self::Rgb(Rgb::from_u8(r, g, b))
	}
	pub const fn hsl_f(h: float, s: float, l: float) -> Self {
		Self::Hsl(Hsl::from_float(h, s, l))
	}
	pub const fn grey(v: float) -> Self {
		Self::Rgb(Rgb::grey(v))
	}
	pub fn new_random() -> Self {
		use typesafe_rng::V2::*;
		let mut rng = thread_rng();
		match rng.gen_typesafe_number() {
			_1 => Self::Hsl(Hsl::new_random()),
			_2 => Self::Rgb(Rgb::new_random()),
		}
	}
	pub fn from_str(s: &str) -> Self {
		let s = s.trim();
		let (format, values) = s.split_at_checked(3).expect("color string is too short, expected at least 3 characters");
		match format {
			"hsl" => Color::Hsl(Hsl::from_str(values)),
			"rgb" => Color::Rgb(Rgb::from_str(values)),
			_ => {
				let maybe_named_color = ALL_COLORS.iter()
					.find(|(name, _color)| name.to_lowercase() == s.to_lowercase());
				if let Some((_name, color)) = maybe_named_color {
					*color
				} else {
					panic!("expected color to be in `rgb` or `hsl` format or to be from list of named colors, but it is `{s}`");
				}
			}
		}
	}
	pub fn to_hsl(self) -> Hsl {
		match self {
			Self::Hsl(hsl) => hsl,
			Self::Rgb(rgb) => rgb.to_hsl(),
		}
	}
	pub fn to_rgb(self) -> Rgb {
		match self {
			Self::Hsl(hsl) => hsl.to_rgb(),
			Self::Rgb(rgb) => rgb,
		}
	}
	pub fn to_image_rgb(self) -> image::Rgb<u8> {
		self.to_rgb().to_image_rgb()
	}
}
impl std::fmt::Display for Color {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let (color_format, v1, v2, v3) = match self {
			Self::Hsl(Hsl { h, s, l }) => ("hsl", h, s, l),
			Self::Rgb(Rgb { r, g, b }) => ("rgb", r, g, b),
		};
		write!(f, "{color_format} {v1} {v2} {v3}")
	}
}

#[derive(Debug, Clone, Copy)]
pub struct Hsl {
	/// 0. ..= 360.
	pub h: float,
	/// 0. ..= 100.
	pub s: float,
	/// 0. ..= 100.
	pub l: float,
}
impl Hsl {
	pub const fn from_float(h: float, s: float, l: float) -> Self {
		assert!(0. <= h && h <= 360.);
		assert!(0. <= s && s <= 100.);
		assert!(0. <= l && l <= 100.);
		Self { h, s, l }
	}
	pub fn new_random() -> Self {
		let mut rng = thread_rng();
		Self::from_float(
			rng.gen_range(0. ..= 360.),
			rng.gen_range(0. ..= 100.),
			rng.gen_range(0. ..= 100.),
		)
	}
	pub fn from_str(s: &str) -> Self {
		todo!()
	}
	pub fn to_rgb(self) -> Rgb {
		let Self { h, s, l } = self;
		let s = s / 100.;
		let l = l / 100.;
		let c = (1. - abs(2.*l - 1.)) * s;
		let ht = h / 60.;
		let x = c * (1. - abs(ht.rem_euclid(2.) - 1.));
		let (r, g, b) = match ht {
			_ if 0. <= ht && ht <= 1. => (c, x, 0.),
			_ if 1. <= ht && ht <= 2. => (x, c, 0.),
			_ if 2. <= ht && ht <= 3. => (0., c, x),
			_ if 3. <= ht && ht <= 4. => (0., x, c),
			_ if 4. <= ht && ht <= 5. => (x, 0., c),
			_ if 5. <= ht && ht <= 6. => (c, 0., x),
			_ => unreachable!()
		};
		Rgb::from_float_in_runtime(r, g, b)
	}
}

#[derive(Debug, Clone, Copy)]
pub struct Rgb {
	/// 0. ..= 1.
	pub r: float,
	/// 0. ..= 1.
	pub g: float,
	/// 0. ..= 1.
	pub b: float,
}
impl Rgb {
	pub const fn from_float_in_consttime(r: float, g: float, b: float) -> Self {
		assert!(0. <= r && r <= 1.);
		assert!(0. <= g && g <= 1.);
		assert!(0. <= b && b <= 1.);
		Self { r, g, b }
	}
	pub fn from_float_in_runtime(r: float, g: float, b: float) -> Self {
		assert!(0. <= r && r <= 1., "r={r}");
		assert!(0. <= g && g <= 1., "g={g}");
		assert!(0. <= b && b <= 1., "b={b}");
		Self { r, g, b }
	}
	pub const fn from_u8(r: u8, g: u8, b: u8) -> Self {
		Self {
			r: r as float / 255.,
			g: g as float / 255.,
			b: b as float / 255.,
		}
	}
	pub const fn grey(v: float) -> Self {
		Self { r: v, g: v, b: v }
	}
	pub fn new_random() -> Self {
		let mut rng = thread_rng();
		Self {
			r: rng.gen_range(0. ..= 1.),
			g: rng.gen_range(0. ..= 1.),
			b: rng.gen_range(0. ..= 1.),
		}
	}
	pub fn from_str(s: &str) -> Self {
		const SPACE: char = ' ';
		const DOT: char = '.';
		let s = s.trim();
		let (r, gb) = s.split_once(SPACE).unwrap();
		let (g, b) = gb.split_once(SPACE).unwrap();
		let r = r.trim();
		let g = g.trim();
		let b = b.trim();
		let r = if r.contains(DOT) { r.parse::<float>().unwrap() } else { r.parse::<u8>().unwrap() as float / 255. };
		let g = if g.contains(DOT) { g.parse::<float>().unwrap() } else { g.parse::<u8>().unwrap() as float / 255. };
		let b = if b.contains(DOT) { b.parse::<float>().unwrap() } else { b.parse::<u8>().unwrap() as float / 255. };
		Self { r, g, b }
	}
	pub fn to_hsl(self) -> Hsl {
		todo!()
	}
	pub fn to_image_rgb(self) -> image::Rgb<u8> {
		image::Rgb([
			(self.r * 255.).round() as u8,
			(self.g * 255.).round() as u8,
			(self.b * 255.).round() as u8,
		])
	}
}


macro_rules! generate_named_colors {
	($(($name:ident, $value:expr)),*) => {
		$(
			pub const $name: Color = $value;
		)*
		pub const ALL_COLORS: &[(&str, Color)] = &[
			$(
				(stringify!($name), $value),
			)*
		];
	};
}

generate_named_colors! [
	(WHITE, Color::rgb_f(1., 1., 1.)),
	(BLACK, Color::rgb_f(0., 0., 0.)),
	(RED  , Color::rgb_f(1., 0., 0.)),
	(GREEN, Color::rgb_f(0., 1., 0.)),
	(BLUE , Color::rgb_f(0., 0., 1.)),
	(CYAN   , Color::rgb_f(0., 1., 1.)),
	(MAGENTA, Color::rgb_f(1., 0., 1.)),
	(YELLOW , Color::rgb_f(1., 1., 0.)),
	(GREY, Color::rgb_f(0.5, 0.5, 0.5)),
	(LIGHT_GREY, Color::rgb_f(0.75, 0.75, 0.75)),
	(DARK_GREY, Color::rgb_f(0.25, 0.25, 0.25)),
	(GREY_10, Color::rgb_f(0.1, 0.1, 0.1)),
	(GREY_20, Color::rgb_f(0.2, 0.2, 0.2)),
	(GREY_30, Color::rgb_f(0.3, 0.3, 0.3)),
	(GREY_40, Color::rgb_f(0.4, 0.4, 0.4)),
	(GREY_50, GREY),
	(GREY_60, Color::rgb_f(0.6, 0.6, 0.6)),
	(GREY_70, Color::rgb_f(0.7, 0.7, 0.7)),
	(GREY_80, Color::rgb_f(0.8, 0.8, 0.8)),
	(GREY_90, Color::rgb_f(0.9, 0.9, 0.9))
];

fn list_all_named_colors() {
	println!("List of all named colors:");
	let longest_name = ALL_COLORS.iter()
		.map(|(name, _value)| name)
		.max_by_key(|name| name.len())
		.unwrap();
	let longest_name_len = longest_name.len();
	for (name, value) in ALL_COLORS {
		let name = name.to_lowercase().replace('_', " ");
		let spaces_n = longest_name_len - name.len() + 1;
		let spaces = " ".repeat(spaces_n);
		println!("- {name}:{spaces}{value}");
	}
}



pub trait ExtensionDateTimeLocalToMyFormat {
	fn to_my_format(&self) -> String;
}
impl ExtensionDateTimeLocalToMyFormat for DateTime<Local> {
	fn to_my_format(&self) -> String {
		let year  : u32 = self.format("%Y").to_string().parse().unwrap();
		let month : u32 = self.format("%m").to_string().parse().unwrap();
		let day   : u32 = self.format("%d").to_string().parse().unwrap();
		let hour  : u32 = self.format("%H").to_string().parse().unwrap();
		let minute: u32 = self.format("%M").to_string().parse().unwrap();
		let second: u32 = self.format("%S").to_string().parse().unwrap();
		let ms    : u32 = self.format("%3f").to_string().parse().unwrap();
		format!("{year:04}-{month:02}-{day:02}_{hour:02}-{minute:02}-{second:02}.{ms:03}")
	}
}




pub fn abs(x: float) -> float { x.abs() }
pub fn sin_raw(x: float) -> float { x.sin() }
pub fn cos_raw(x: float) -> float { x.cos() }
pub fn sqrt(x: float) -> float { x.sqrt() }


