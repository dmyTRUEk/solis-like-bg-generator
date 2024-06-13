//! Main file, yeah
//!
//! Yes, it's full of magick numbers. Hope you enjoy them! ;)

#![feature(
	const_fn_floating_point_arithmetic,
)]

use chrono::{DateTime, Local};
use clap::{Parser, ValueEnum};
use image::ImageBuffer;
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Poisson};

mod float_type;
mod typesafe_rng;

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
	#[arg(short='a', long, default_value_t=150.)]
	angle_of_rays: float,

	#[arg(long, default_value_t=NoiseType::Mul)]
	pre_noise_type: NoiseType,
	#[arg(long, default_value_t=1.1)]
	pre_noise_value: float,

	#[arg(long, default_value_t=NoiseType::Mul)]
	mid_noise_type: NoiseType,
	#[arg(long, default_value_t=1.1)]
	mid_noise_value: float,
	#[arg(long, default_value_t=0.1)]
	mid_noise_probability: float,

	#[arg(long, default_value_t=NoiseType::Mul)]
	post_noise_type: NoiseType,
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

	angle_of_rays: float,

	pre_noise: Option<Noise>,
	mid_noise: Option<(Noise, float)>,
	post_noise: Option<Noise>,
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

			angle_of_rays: angle_of_rays.rem_euclid(360.),

			pre_noise: Noise::from_type_and_value(pre_noise_type, pre_noise_value),
			mid_noise: Noise::from_type_and_value_and_prob(mid_noise_type, mid_noise_value, mid_noise_probability),
			post_noise: Noise::from_type_and_value(post_noise_type, post_noise_value),
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

	let mut rng = thread_rng();

	// img.fill_by_noise_grey(SinNoise::new_random_default());
	// img.fill_by_noise_rgb(
	// 	SinNoise::new_random_default(),
	// 	SinNoise::new_random_default(),
	// 	SinNoise::new_random_default(),
	// );

	let image_wh_f = (img.width() as float, img.height() as float);
	// let mut rpn = RandomPointsNoise::new_random_default(image_wh_f);
	// rpn.metric = Metric::GeometricMean;
	// img.fill_by_noise_grey(rpn);
	img.fill_by_noise_rgb(
		RandomPointsNoise::new_random_default(image_wh_f),
		RandomPointsNoise::new_random_default(image_wh_f),
		RandomPointsNoise::new_random_default(image_wh_f),
	);

	// img.fill_by_color(cli_args.background_color);
	// // if let Some(pre_noise) = cli_args.pre_noise {
	// // 	img.apply(pre_noise);
	// // }

	// let shapes_n: usize = rng.sample(Poisson::new(4_f32).unwrap()).round() as usize;
	// let mut shapes = Vec::<ShapeWithColor>::with_capacity(shapes_n);
	// for _ in 0..shapes_n {
	// 	shapes.push(ShapeWithColor {
	// 		shape: Shape::new_random(cli_args.image_w, cli_args.image_h),
	// 		color: Color::new_random(),
	// 	});
	// }
	// for shape in shapes {
	// 	img.apply(shape);
	// }

	// img.apply(PullFromWithAngle {
	// 	pull_from: PullFrom::AllArea,
	// 	angle: cli_args.angle_of_rays,
	// });

	let filename = format!("slb_{now}.png", now=timestamp_program_start.to_my_format());
	img.save(filename).expect("unable to save image")
}


trait ImgMethods {
	fn fill_by_color(&mut self, color: Color);
	fn fill_by_noise_grey(&mut self, noise: impl NoiseFns);
	fn fill_by_noise_rgb(&mut self, noise_r: impl NoiseFns, noise_g: impl NoiseFns, noise_b: impl NoiseFns);
}
impl ImgMethods for ImageBuf {
	fn fill_by_color(&mut self, color: Color) {
		let color = color.to_image_rgb();
		for pixel in self.pixels_mut() {
			*pixel = color;
		}
	}
	fn fill_by_noise_grey(&mut self, noise: impl NoiseFns) {
		let w_half = self.width()  as float / 2.;
		let h_half = self.height() as float / 2.;
		for (x, y, pixel) in self.enumerate_pixels_mut() {
			*pixel = Color::grey(noise.at(x as float - w_half, y as float - h_half)).to_image_rgb();
		}
	}
	fn fill_by_noise_rgb(&mut self, noise_r: impl NoiseFns, noise_g: impl NoiseFns, noise_b: impl NoiseFns) {
		for (x, y, pixel) in self.enumerate_pixels_mut() {
			*pixel = Color::rgb_f(
				noise_r.at(x as float, y as float),
				noise_g.at(x as float, y as float),
				noise_b.at(x as float, y as float),
			).to_image_rgb();
		}
	}
}


impl Apply<Noise> for ImageBuf {
	fn apply(&mut self, noise: Noise) {
		for pixel in self.pixels_mut() {
			pixel.apply(noise);
		}
	}
}


pub enum Shape {
	Rectangle { x: float, y: float, w: float, h: float },
}
impl Shape {
	pub fn new_random(image_w: u32, image_h: u32) -> Self {
		let mut rng = thread_rng();
		let image_w = image_w as float;
		let image_h = image_h as float;
		let image_w_half = image_w / 2.;
		let image_h_half = image_h / 2.;
		use typesafe_rng::V1::*;
		match rng.gen_typesafe_number() {
			_0 => Self::Rectangle {
				x: rng.gen_range(-image_w_half ..= image_w_half),
				y: rng.gen_range(-image_h_half ..= image_h_half),
				w: rng.sample(Normal::new(image_w_half/2., image_w_half/2.).unwrap()).abs(),
				h: rng.sample(Normal::new(image_h_half/2., image_h_half/2.).unwrap()).abs(),
			}
		}
	}
	pub fn contains(&self, px: float, py: float) -> bool {
		match self {
			Self::Rectangle { x, y, w, h } => {
				x-w/2. <= px && px <= x+w/2. &&
				y-h/2. <= py && py <= y+h/2.
			}
		}
	}
	pub fn is_on_perimeter(&self, px: float, py: float, delta: float) -> bool {
		match self {
			Self::Rectangle { x, y, w, h } => {
				abs(x-w/2.-px) <= delta || abs(x+w/2.-px) <= delta ||
				abs(y-h/2.-py) <= delta || abs(y+h/2.-py) <= delta
			},
		}
	}
}

pub struct ShapeWithColor {
	shape: Shape,
	color: Color,
}

impl Apply<ShapeWithColor> for ImageBuf {
	fn apply(&mut self, ShapeWithColor { shape, color }: ShapeWithColor) {
		let color = color.to_image_rgb();
		let w_half = self.width()  as float / 2.;
		let h_half = self.height() as float / 2.;
		// TODO(optimization): loop only from min to max
		for (px, py, pixel) in self.enumerate_pixels_mut() {
			if shape.contains(px as float - w_half, py as float - h_half) {
				*pixel = color;
			}
		}
	}
}


pub enum PullFrom {
	AllArea,
	Shape(Shape),
	ShapePerimeter(Shape),
}
pub struct PullFromWithAngle {
	pull_from: PullFrom,
	angle: float,
}
impl Apply<PullFromWithAngle> for ImageBuf {
	fn apply(&mut self, PullFromWithAngle { pull_from, angle }: PullFromWithAngle) {
		let w_half = self.width()  as float / 2.;
		let h_half = self.height() as float / 2.;
		match pull_from {
			PullFrom::AllArea => {
				match angle {
					_ if 0. <= angle && angle <= 90. => {
						// top right corner
						todo!()
					}
					_ if 90. <= angle && angle <= 180. => {
						// top left corner
						todo!()
					}
					_ if 180. <= angle && angle <= 270. => {
						// bottom left corner
						todo!()
					}
					_ if 270. <= angle && angle <= 360. => {
						// bottom right corner
						for px in 0..self.width() {
							for py in 0..self.height() {
								self[(px, py)] = todo!();
							}
						}
					}
					_ => unreachable!()
				}
			}
			PullFrom::Shape(shape) => {
				todo!();
				// for (px, py, pixel) in self.enumerate_pixels_mut() {
				// 	if shape.contains(px as float - w_half, py as float - h_half) {
				// 		*pixel = color;
				// 	}
				// }
			}
			PullFrom::ShapePerimeter(shape) => {
				todo!()
			}
		}
	}
}


pub trait NoiseFns {
	fn at(&self, x: float, y: float) -> float;
}


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
		let value_from_x: float = self.frequencies_x.iter().map(|f| (sin(f*x-f)+1.)/2.).sum();
		let value_from_y: float = self.frequencies_y.iter().map(|f| (sin(f*y-f)+1.)/2.).sum();
		(value_from_x + value_from_y) / (self.frequencies_x.len() + self.frequencies_y.len()) as float
	}
	pub fn prod_at(&self, x: float, y: float) -> float {
		let value_from_x: float = self.frequencies_x.iter().map(|f| (sin(f*x-f)+1.)/2.).product();
		let value_from_y: float = self.frequencies_y.iter().map(|f| (sin(f*y-f)+1.)/2.).product();
		value_from_x * value_from_y
	}
	pub fn prod_sqrt_at(&self, x: float, y: float) -> float {
		self.prod_at(x, y).powf(1. / (self.frequencies_x.len() + self.frequencies_y.len()) as float)
	}
}
impl NoiseFns for SinNoise {
	fn at(&self, x: float, y: float) -> float {
		// self.sum_at(x, y)
		// self.prod_at(x, y)
		self.prod_sqrt_at(x, y)
		// (self.sum_at(x, y) + self.prod_at(x, y)) / 2.
	}
}


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
impl NoiseFns for RandomPointsNoise {
	fn at(&self, x: float, y: float) -> float {
		let dist_to_closest_point: float = self.points.iter()
			.map(|&(px, py)| self.metric.dist((px, py), (x, y)))
			.min_by(|d1, d2| d1.total_cmp(d2))
			.unwrap();
		(dist_to_closest_point / self.avg_dist).clamp(0., 1.)
	}
}


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
			_0 => Self::Abs,
			_1 => Self::Euclid,
			_2 => Self::Pow(3.+rng.sample(Poisson::new(0.7).unwrap())),
			_3 => Self::GeometricMean,
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
enum NoiseType {
	None,
	Add,
	Mul,
}
impl ToString for NoiseType {
	fn to_string(&self) -> String {
		match self {
			Self::None => format!("none"),
			Self::Add  => format!("add"),
			Self::Mul  => format!("mul"),
		}
	}
}

#[derive(Debug, Clone, Copy)]
enum Noise {
	Add { value: float },
	Mul { value: float },
}
impl Noise {
	fn from_type_and_value(type_: NoiseType, value: float) -> Option<Self> {
		match type_ {
			NoiseType::None => None,
			NoiseType::Add => Some(Self::Add { value }),
			NoiseType::Mul => Some(Self::Mul { value }),
		}
	}
	fn from_type_and_value_and_prob(type_: NoiseType, value: float, prob: float) -> Option<(Self, float)> {
		Self::from_type_and_value(type_, value)
			.map(|n| (n, prob))
	}
}
trait Apply<T> {
	fn apply(&mut self, t: T);
}
impl Apply<Noise> for image::Rgb<u8> {
	fn apply(&mut self, noise: Noise) {
	    self.0[0].apply(noise);
	    self.0[1].apply(noise);
	    self.0[2].apply(noise);
	}
}
impl Apply<Noise> for u8 {
	fn apply(&mut self, noise: Noise) {
		*self = match noise {
			Noise::Add { value } => {
				let random_value: float = thread_rng().gen_range(-value ..= value);
				((*self as float / 255. + random_value) * 255.).round().clamp(0., 255.) as u8
			}
			Noise::Mul { value } => {
				let v1 = value;
				let v2 = 1. / value;
				let v_min = v1.min(v2);
				let v_max = v1.max(v2);
				let random_value: float = thread_rng().gen_range(v_min ..= v_max);
				(*self as float * random_value).round().clamp(0., 255.) as u8
			}
		}
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
			_0 => Self::Hsl(Hsl::new_random()),
			_1 => Self::Rgb(Rgb::new_random()),
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
pub fn sin(x: float) -> float { x.sin() }
pub fn cos(x: float) -> float { x.cos() }
pub fn sqrt(x: float) -> float { x.sqrt() }


