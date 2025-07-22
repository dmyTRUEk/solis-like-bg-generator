//! Wrappers for angle in deg and rad to not confuse them.

use rand::{thread_rng, Rng};

use crate::float_type::float;

#[derive(Debug, Clone, Copy)]
pub struct AngleDeg(float);
#[derive(Debug, Clone, Copy)]
pub struct AngleRad(float);

pub enum Octant { _1, _2, _3, _4, _5, _6, _7, _8 }

impl From<AngleDeg> for AngleRad {
	fn from(angle_in_deg: AngleDeg) -> AngleRad {
		AngleRad(angle_in_deg.0.to_radians())
	}
}
impl From<AngleRad> for AngleDeg {
	fn from(angle_in_rad: AngleRad) -> AngleDeg {
		AngleDeg(angle_in_rad.0.to_degrees())
	}
}

impl AngleDeg {
	pub fn new(degrees: float) -> AngleDeg {
		// assert!(0. <= degrees && degrees <= 360., "angle in degrees must be between 0 and 360, but it is {degrees}");
		AngleDeg(degrees.rem_euclid(360.))
	}
	pub fn new_random() -> AngleDeg {
		let mut rng = thread_rng();
		AngleDeg::new(rng.gen_range(0. ..= 360.))
	}
	pub fn cos(self) -> float {
		AngleRad::from(self).0.cos()
	}
	pub fn sin(self) -> float {
		AngleRad::from(self).0.sin()
	}
	pub fn octant(self) -> Octant {
		match self {
			_ if self.is_oct1() => Octant::_1,
			_ if self.is_oct2() => Octant::_2,
			_ if self.is_oct3() => Octant::_3,
			_ if self.is_oct4() => Octant::_4,
			_ if self.is_oct5() => Octant::_5,
			_ if self.is_oct6() => Octant::_6,
			_ if self.is_oct7() => Octant::_7,
			_ if self.is_oct8() => Octant::_8,
			_ => unreachable!()
		}
	}
	pub fn is_oct1(self) -> bool { 0. <= self.0 && self.0 <= 45. }
	pub fn is_oct2(self) -> bool { 45. <= self.0 && self.0 <= 90. }
	pub fn is_oct3(self) -> bool { 90. <= self.0 && self.0 <= 135. }
	pub fn is_oct4(self) -> bool { 135. <= self.0 && self.0 <= 180. }
	pub fn is_oct5(self) -> bool { 180. <= self.0 && self.0 <= 225. }
	pub fn is_oct6(self) -> bool { 225. <= self.0 && self.0 <= 270. }
	pub fn is_oct7(self) -> bool { 270. <= self.0 && self.0 <= 315. }
	pub fn is_oct8(self) -> bool { 315. <= self.0 && self.0 <= 360. }
}

