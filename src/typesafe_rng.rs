//! RNG utils.

use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};


pub trait TypesafeRNG<const N: usize, T> {
	fn gen_typesafe_number(&mut self) -> T;
	fn gen_typesafe_with_weights(&mut self, weights: [u32; N]) -> T;
}

macro_rules! impl_gen_with_weights {
	($num:literal, $name:ident, $elems:tt) => {
		#[derive(Debug, Clone, Copy, IntoPrimitive, TryFromPrimitive)]
		#[repr(u8)]
		pub enum $name $elems
		impl<R: Rng> TypesafeRNG<$num, $name> for R {
			fn gen_typesafe_number(&mut self) -> $name {
				let n: u8 = self.gen_range(0..$num);
				$name::try_from(n).unwrap()
			}
			fn gen_typesafe_with_weights(&mut self, weights: [u32; $num]) -> $name {
				let n = WeightedIndex::new(&weights).unwrap().sample(self);
				// `u8` because #[repr(u8)]
				let n: u8 = n.try_into().unwrap();
				$name::try_from(n).unwrap()
			}
		}
	}
}

// TODO: somehow use `cargo expand` to see the output of only this file/macro?
impl_gen_with_weights!(1, V1, { _0 });
impl_gen_with_weights!(2, V2, { _0, _1 });
impl_gen_with_weights!(3, V3, { _0, _1, _2 });
impl_gen_with_weights!(4, V4, { _0, _1, _2, _3 });
impl_gen_with_weights!(5, V5, { _0, _1, _2, _3, _4 });
impl_gen_with_weights!(6, V6, { _0, _1, _2, _3, _4, _5 });
impl_gen_with_weights!(7, V7, { _0, _1, _2, _3, _4, _5, _6 });
impl_gen_with_weights!(8, V8, { _0, _1, _2, _3, _4, _5, _6, _7 });
impl_gen_with_weights!(9, V9, { _0, _1, _2, _3, _4, _5, _6, _7, _8 });

