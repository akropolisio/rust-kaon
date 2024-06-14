// SPDX-License-Identifier: CC0-1.0

//! Bitcoin/Kaon amounts.
//!
//! This module mainly introduces the [Amount] and [SignedAmount] types.
//! We refer to the documentation on the types for more information.
//! Important:
//! In order to avoid any confusion every btc mention was changed to KAON.
//! If that'll create a problem for developers, we may create series with aliases.

#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};
use core::cmp::Ordering;
#[cfg(feature = "alloc")]
use core::fmt::Write as _;
use core::str::FromStr;
use core::{default, fmt, ops};

#[cfg(feature = "serde")]
use ::serde::{Deserialize, Serialize};
use internals::error::InputString;
use internals::write_err;

/// A set of denominations in which amounts can be expressed.
/// msats, ustats, nsats and psats are described in
/// BOLT11, see: https://github.com/lightning/bolts/blob/master/11-payment-encoding.md
///
/// Added prefix Atto- for the lowest denomination: AttoKAON(aKAON) or 1e-18 KAON
/// # Examples
///
/// ```
/// # use core::str::FromStr;
/// # use kaon_units::Amount;
///
/// assert_eq!(Amount::from_str("1 KAON").unwrap(), Amount::from_sat(1_000_000_000_000_000_000));
/// assert_eq!(Amount::from_str("1 cKAON").unwrap(), Amount::from_sat(10_000_000_000_000_000));
/// assert_eq!(Amount::from_str("1 mKAON").unwrap(), Amount::from_sat(1_000_000_000_000_000));
/// assert_eq!(Amount::from_str("1 uKAON").unwrap(), Amount::from_sat(1_000_000_000_000));
/// assert_eq!(Amount::from_str("1 nKAON").unwrap(), Amount::from_sat(1_000_000_000));
/// assert_eq!(Amount::from_str("1 pKAON").unwrap(), Amount::from_sat(1_000_000));
/// assert_eq!(Amount::from_str("1 aKAON").unwrap(), Amount::from_sat(1));
/// assert_eq!(Amount::from_str("1 bit").unwrap(), Amount::from_sat(1_000_000_000_000));
/// assert_eq!(Amount::from_str("1 sat").unwrap(), Amount::from_sat(10_000_000_000));
/// assert_eq!(Amount::from_str("1 msats").unwrap(), Amount::from_sat(10_000_000));
/// assert_eq!(Amount::from_str("1 usats").unwrap(), Amount::from_sat(10_000));
/// assert_eq!(Amount::from_str("1 nsats").unwrap(), Amount::from_sat(10));
/// assert_eq!(Amount::from_str("100 psats").unwrap(), Amount::from_sat(1));
/// ```
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Denomination {
    /// KAON
    KAON,
    /// cKAON
    CentiKAON,
    /// mKAON
    MilliKAON,
    /// uKAON
    MicroKAON,
    /// nKAON
    NanoKAON,
    /// pKAON
    PicoKAON,
    /// aKAON
    AttoKAON,
    /// bits
    Bit,
    /// satoshi
    Satoshi,
    /// msat
    MilliSatoshi,
    /// usat
    MicroSatoshi,
    /// nsat
    NanoSatoshi,
    /// psat
    PicoSatoshi,
}

impl Denomination {
    /// Convenience alias for `Denomination::KAON`.
    pub const KAON: Self = Denomination::KAON;

    /// Convenience alias for `Denomination::Satoshi`.
    pub const SAT: Self = Denomination::Satoshi;

    /// Convenience alias for `Denomination::AttoKAON`.
    pub const A_KAON: Self = Denomination::AttoKAON;

    /// The number of decimal places more than AttoKAON.
    fn precision(self) -> i8 {
        match self {
            Denomination::KAON => -18,
            Denomination::CentiKAON => -16,
            Denomination::MilliKAON => -15,
            Denomination::MicroKAON => -12,
            Denomination::NanoKAON => -9,
            Denomination::PicoKAON => -6,
            Denomination::AttoKAON => 0,
            Denomination::Bit => -12,
            Denomination::Satoshi => -10,
            Denomination::MilliSatoshi => -7,
            Denomination::MicroSatoshi => -4,
            Denomination::NanoSatoshi => -1,
            Denomination::PicoSatoshi => 2,
        }
    }

    /// Returns stringly representation of this
    fn as_str(self) -> &'static str {
        match self {
            Denomination::KAON => "KAON",
            Denomination::CentiKAON => "cKAON",
            Denomination::MilliKAON => "mKAON",
            Denomination::MicroKAON => "uKAON",
            Denomination::NanoKAON => "nKAON",
            Denomination::PicoKAON => "pKAON",
            Denomination::AttoKAON => "aKAON",
            Denomination::Bit => "bits",
            Denomination::Satoshi => "satoshi",
            Denomination::MilliSatoshi => "msat",
            Denomination::MicroSatoshi => "usat",
            Denomination::NanoSatoshi => "nsat",
            Denomination::PicoSatoshi => "psat",
        }
    }

    /// The different str forms of denominations that are recognized.
    fn forms(s: &str) -> Option<Self> {
        match s {
            "KAON" | "kaon" => Some(Denomination::KAON),
            "cKAON" | "ckaon" => Some(Denomination::CentiKAON),
            "mKAON" | "mkaon" => Some(Denomination::MilliKAON),
            "uKAON" | "ukaon" => Some(Denomination::MicroKAON),
            "nKAON" | "nkaon" => Some(Denomination::NanoKAON),
            "pKAON" | "pkaon" => Some(Denomination::PicoKAON),
            "aKAON" | "akaon" => Some(Denomination::AttoKAON),
            "bit" | "bits" | "BIT" | "BITS" => Some(Denomination::Bit),
            "SATOSHI" | "satoshi" | "SATOSHIS" | "satoshis" | "SAT" | "sat" | "SATS" | "sats" =>
                Some(Denomination::Satoshi),
            "mSAT" | "msat" | "mSATs" | "msats" => Some(Denomination::MilliSatoshi),
            "uSAT" | "usat" | "uSATs" | "usats" => Some(Denomination::MicroSatoshi),
            "nSAT" | "nsat" | "nSATs" | "nsats" => Some(Denomination::NanoSatoshi),
            "pSAT" | "psat" | "pSATs" | "psats" | "picosat" | "picosats" =>
                Some(Denomination::PicoSatoshi),
            _ => None,
        }
    }
}

/// These form are ambigous and could have many meanings.  For example, M could denote Mega or Milli.
/// If any of these forms are used, an error type PossiblyConfusingDenomination is returned.
const CONFUSING_FORMS: [&str; 9] =
    ["Msat", "Msats", "MSAT", "MSATS", "MSat", "MSats", "MKAON", "Mkaon", "PKAON"];

impl fmt::Display for Denomination {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str(self.as_str()) }
}

impl FromStr for Denomination {
    type Err = ParseDenominationError;

    /// Convert from a str to Denomination.
    ///
    /// Any combination of upper and/or lower case, excluding uppercase of SI(m, u, n, p, a) is considered valid.
    /// - Singular: KAON, mKAON, uKAON, nKAON, pKAON, aKAON
    /// - Plural or singular: sat, satoshi, bit, msat
    ///
    /// Due to ambiguity between mega and milli, pico and peta we prohibit usage of leading capital 'M', 'P'.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use self::ParseDenominationError::*;

        if CONFUSING_FORMS.contains(&s) {
            return Err(PossiblyConfusing(PossiblyConfusingDenominationError(s.into())));
        };

        let form = self::Denomination::forms(s);

        form.ok_or_else(|| Unknown(UnknownDenominationError(s.into())))
    }
}

/// An error during amount parsing amount with denomination.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseError {
    /// Invalid amount.
    Amount(ParseAmountError),

    /// Invalid denomination.
    Denomination(ParseDenominationError),

    /// The denomination was not identified.
    MissingDenomination(MissingDenominationError),
}

internals::impl_from_infallible!(ParseError);

impl From<ParseAmountError> for ParseError {
    fn from(e: ParseAmountError) -> Self { Self::Amount(e) }
}

impl From<ParseDenominationError> for ParseError {
    fn from(e: ParseDenominationError) -> Self { Self::Denomination(e) }
}

impl From<OutOfRangeError> for ParseError {
    fn from(e: OutOfRangeError) -> Self { Self::Amount(e.into()) }
}

impl From<TooPreciseError> for ParseError {
    fn from(e: TooPreciseError) -> Self { Self::Amount(e.into()) }
}

impl From<MissingDigitsError> for ParseError {
    fn from(e: MissingDigitsError) -> Self { Self::Amount(e.into()) }
}

impl From<InputTooLargeError> for ParseError {
    fn from(e: InputTooLargeError) -> Self { Self::Amount(e.into()) }
}

impl From<InvalidCharacterError> for ParseError {
    fn from(e: InvalidCharacterError) -> Self { Self::Amount(e.into()) }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParseError::Amount(error) => write_err!(f, "invalid amount"; error),
            ParseError::Denomination(error) => write_err!(f, "invalid denomination"; error),
            // We consider this to not be a source because it currently doesn't contain useful
            // information
            ParseError::MissingDenomination(_) =>
                f.write_str("the input doesn't contain a denomination"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParseError::Amount(error) => Some(error),
            ParseError::Denomination(error) => Some(error),
            // We consider this to not be a source because it currently doesn't contain useful
            // information
            ParseError::MissingDenomination(_) => None,
        }
    }
}

/// An error during amount parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseAmountError {
    /// The amount is too big or too small.
    OutOfRange(OutOfRangeError),
    /// Amount has higher precision than supported by the type.
    TooPrecise(TooPreciseError),
    /// A digit was expected but not found.
    MissingDigits(MissingDigitsError),
    /// Input string was too large.
    InputTooLarge(InputTooLargeError),
    /// Invalid character in input.
    InvalidCharacter(InvalidCharacterError),
}

impl From<TooPreciseError> for ParseAmountError {
    fn from(value: TooPreciseError) -> Self { Self::TooPrecise(value) }
}

impl From<MissingDigitsError> for ParseAmountError {
    fn from(value: MissingDigitsError) -> Self { Self::MissingDigits(value) }
}

impl From<InputTooLargeError> for ParseAmountError {
    fn from(value: InputTooLargeError) -> Self { Self::InputTooLarge(value) }
}

impl From<InvalidCharacterError> for ParseAmountError {
    fn from(value: InvalidCharacterError) -> Self { Self::InvalidCharacter(value) }
}

internals::impl_from_infallible!(ParseAmountError);

impl fmt::Display for ParseAmountError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ParseAmountError::*;

        match *self {
            OutOfRange(ref error) => write_err!(f, "amount out of range"; error),
            TooPrecise(ref error) => write_err!(f, "amount has a too high precision"; error),
            MissingDigits(ref error) => write_err!(f, "the input has too few digits"; error),
            InputTooLarge(ref error) => write_err!(f, "the input is too large"; error),
            InvalidCharacter(ref error) => write_err!(f, "invalid character in the input"; error),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseAmountError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use ParseAmountError::*;

        match *self {
            TooPrecise(ref error) => Some(error),
            InputTooLarge(ref error) => Some(error),
            OutOfRange(ref error) => Some(error),
            MissingDigits(ref error) => Some(error),
            InvalidCharacter(ref error) => Some(error),
        }
    }
}

/// Returned when a parsed amount is too big or too small.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OutOfRangeError {
    is_signed: bool,
    is_greater_than_max: bool,
}

impl OutOfRangeError {
    /// Returns the minimum and maximum allowed values for the type that was parsed.
    ///
    /// This can be used to give a hint to the user which values are allowed.
    pub fn valid_range(&self) -> (i128, u128) {
        match self.is_signed {
            true => (i128::MIN, i128::MAX as u128),
            false => (0, u128::MAX),
        }
    }

    /// Returns true if the input value was large than the maximum allowed value.
    pub fn is_above_max(&self) -> bool { self.is_greater_than_max }

    /// Returns true if the input value was smaller than the minimum allowed value.
    pub fn is_below_min(&self) -> bool { !self.is_greater_than_max }

    pub(crate) fn too_big(is_signed: bool) -> Self { Self { is_signed, is_greater_than_max: true } }

    pub(crate) fn too_small() -> Self {
        Self {
            // implied - negative() is used for the other
            is_signed: true,
            is_greater_than_max: false,
        }
    }

    pub(crate) fn negative() -> Self {
        Self {
            // implied - too_small() is used for the other
            is_signed: false,
            is_greater_than_max: false,
        }
    }
}

impl fmt::Display for OutOfRangeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_greater_than_max {
            write!(f, "the amount is greater than {}", self.valid_range().1)
        } else {
            write!(f, "the amount is less than {}", self.valid_range().0)
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OutOfRangeError {}

impl From<OutOfRangeError> for ParseAmountError {
    fn from(value: OutOfRangeError) -> Self { ParseAmountError::OutOfRange(value) }
}

/// Error returned when the input string has higher precision than AttoKAON.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TooPreciseError {
    position: usize,
}

impl fmt::Display for TooPreciseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.position {
            0 => f.write_str("the amount is less than 1 satoshi but it's not zero"),
            pos => write!(
                f,
                "the digits starting from position {} represent a sub-satoshi amount",
                pos
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TooPreciseError {}

/// Error returned when the input string is too large.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InputTooLargeError {
    len: usize,
}

impl fmt::Display for InputTooLargeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.len - INPUT_STRING_LEN_LIMIT {
            1 => write!(
                f,
                "the input is one character longer than the maximum allowed length ({})",
                INPUT_STRING_LEN_LIMIT
            ),
            n => write!(
                f,
                "the input is {} characters longer than the maximum allowed length ({})",
                n, INPUT_STRING_LEN_LIMIT
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InputTooLargeError {}

/// Error returned when digits were expected in the input but there were none.
///
/// In particular, this is currently returned when the string is empty or only contains the minus sign.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MissingDigitsError {
    kind: MissingDigitsKind,
}

impl fmt::Display for MissingDigitsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.kind {
            MissingDigitsKind::Empty => f.write_str("the input is empty"),
            MissingDigitsKind::OnlyMinusSign =>
                f.write_str("there are no digits following the minus (-) sign"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MissingDigitsError {}

#[derive(Debug, Clone, Eq, PartialEq)]
enum MissingDigitsKind {
    Empty,
    OnlyMinusSign,
}

/// Returned when the input contains an invalid character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidCharacterError {
    invalid_char: char,
    position: usize,
}

impl fmt::Display for InvalidCharacterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.invalid_char {
            '.' => f.write_str("there is more than one decimal separator (dot) in the input"),
            '-' => f.write_str("there is more than one minus sign (-) in the input"),
            c => write!(
                f,
                "the character '{}' at position {} is not a valid digit",
                c, self.position
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InvalidCharacterError {}

/// An error during amount parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseDenominationError {
    /// The denomination was unknown.
    Unknown(UnknownDenominationError),
    /// The denomination has multiple possible interpretations.
    PossiblyConfusing(PossiblyConfusingDenominationError),
}

internals::impl_from_infallible!(ParseDenominationError);

impl fmt::Display for ParseDenominationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ParseDenominationError::*;

        match *self {
            Unknown(ref e) => write_err!(f, "denomination parse error"; e),
            PossiblyConfusing(ref e) => write_err!(f, "denomination parse error"; e),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseDenominationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use ParseDenominationError::*;

        match *self {
            Unknown(_) | PossiblyConfusing(_) => None,
        }
    }
}

/// Error returned when the denomination is empty.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MissingDenominationError;

/// Parsing error, unknown denomination.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnknownDenominationError(InputString);

impl fmt::Display for UnknownDenominationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.unknown_variant("KAON denomination", f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UnknownDenominationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}

/// Parsing error, possibly confusing denomination.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PossiblyConfusingDenominationError(InputString);

impl fmt::Display for PossiblyConfusingDenominationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: possibly confusing denomination - we intentionally do not support 'M' and 'P' so as to not confuse mega/milli and peta/pico", self.0.display_cannot_parse("KAON denomination"))
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PossiblyConfusingDenominationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}

/// Returns `Some(position)` if the precision is not supported.
///
/// The position indicates the first digit that is too precise.
fn is_too_precise(s: &str, precision: usize) -> Option<usize> {
    match s.find('.') {
        Some(pos) if precision >= pos => Some(0),
        Some(pos) => s[..pos]
            .char_indices()
            .rev()
            .take(precision)
            .find(|(_, d)| *d != '0')
            .map(|(i, _)| i)
            .or_else(|| {
                s[(pos + 1)..].char_indices().find(|(_, d)| *d != '0').map(|(i, _)| i + pos + 1)
            }),
        None if precision >= s.len() => Some(0),
        None => s.char_indices().rev().take(precision).find(|(_, d)| *d != '0').map(|(i, _)| i),
    }
}

const INPUT_STRING_LEN_LIMIT: usize = 50;

/// Parse decimal string in the given denomination into a AttoKAON value and a
/// bool indicator for a negative amount.
fn parse_signed_to_satoshi(
    mut s: &str,
    denom: Denomination,
) -> Result<(bool, u128), InnerParseError> {
    if s.is_empty() {
        return Err(InnerParseError::MissingDigits(MissingDigitsError {
            kind: MissingDigitsKind::Empty,
        }));
    }
    if s.len() > INPUT_STRING_LEN_LIMIT {
        return Err(InnerParseError::InputTooLarge(s.len()));
    }

    // TODO: add proper support for negative int128
    let is_negative = s.starts_with('-');
    if is_negative {
        if s.len() == 1 {
            return Err(InnerParseError::MissingDigits(MissingDigitsError {
                kind: MissingDigitsKind::OnlyMinusSign,
            }));
        }
        s = &s[1..];
    }

    let max_decimals = {
        // The difference in precision between native (AttoKAON)
        // and desired denomination.
        let precision_diff = -denom.precision();
        if precision_diff <= 0 {
            // If precision diff is negative, this means we are parsing
            // into a less precise amount. That is not allowed unless
            // there are no decimals and the last digits are zeroes as
            // many as the difference in precision.
            let last_n = precision_diff.unsigned_abs().into();
            if let Some(position) = is_too_precise(s, last_n) {
                match s.parse::<i128>() {
                    Ok(0) => return Ok((is_negative, 0)),
                    _ =>
                        return Err(InnerParseError::TooPrecise(TooPreciseError {
                            position: position + is_negative as usize,
                        })),
                }
            }
            s = &s[0..s.find('.').unwrap_or(s.len()) - last_n];
            0
        } else {
            precision_diff
        }
    };

    let mut decimals = None;
    let mut value: u128 = 0; // as AttoKAON
    for (i, c) in s.char_indices() {
        match c {
            '0'..='9' => {
                // Do `value = 10 * value + digit`, catching overflows.
                match 10_u128.checked_mul(value) {
                    None => return Err(InnerParseError::Overflow { is_negative }),
                    Some(val) => match val.checked_add((c as u8 - b'0') as u128) {
                        None => return Err(InnerParseError::Overflow { is_negative }),
                        Some(val) => value = val,
                    },
                }
                // Increment the decimal digit counter if past decimal.
                decimals = match decimals {
                    None => None,
                    Some(d) if d < max_decimals => Some(d + 1),
                    _ =>
                        return Err(InnerParseError::TooPrecise(TooPreciseError {
                            position: i + is_negative as usize,
                        })),
                };
            }
            '.' => match decimals {
                None if max_decimals <= 0 => break,
                None => decimals = Some(0),
                // Double decimal dot.
                _ =>
                    return Err(InnerParseError::InvalidCharacter(InvalidCharacterError {
                        invalid_char: '.',
                        position: i + is_negative as usize,
                    })),
            },
            c =>
                return Err(InnerParseError::InvalidCharacter(InvalidCharacterError {
                    invalid_char: c,
                    position: i + is_negative as usize,
                })),
        }
    }

    // Decimally shift left by `max_decimals - decimals`.
    let scale_factor = max_decimals - decimals.unwrap_or(0);
    for _ in 0..scale_factor {
        value = match 10_u128.checked_mul(value) {
            Some(v) => v,
            None => return Err(InnerParseError::Overflow { is_negative }),
        };
    }

    Ok((is_negative, value))
}

enum InnerParseError {
    Overflow { is_negative: bool },
    TooPrecise(TooPreciseError),
    MissingDigits(MissingDigitsError),
    InputTooLarge(usize),
    InvalidCharacter(InvalidCharacterError),
}

internals::impl_from_infallible!(InnerParseError);

impl InnerParseError {
    fn convert(self, is_signed: bool) -> ParseAmountError {
        match self {
            Self::Overflow { is_negative } =>
                OutOfRangeError { is_signed, is_greater_than_max: !is_negative }.into(),
            Self::TooPrecise(error) => ParseAmountError::TooPrecise(error),
            Self::MissingDigits(error) => ParseAmountError::MissingDigits(error),
            Self::InputTooLarge(len) => ParseAmountError::InputTooLarge(InputTooLargeError { len }),
            Self::InvalidCharacter(error) => ParseAmountError::InvalidCharacter(error),
        }
    }
}

fn split_amount_and_denomination(s: &str) -> Result<(&str, Denomination), ParseError> {
    let (i, j) = if let Some(i) = s.find(' ') {
        (i, i + 1)
    } else {
        let i = s
            .find(|c: char| c.is_alphabetic())
            .ok_or(ParseError::MissingDenomination(MissingDenominationError))?;
        (i, i)
    };
    Ok((&s[..i], s[j..].parse()?))
}

/// Options given by `fmt::Formatter`
struct FormatOptions {
    fill: char,
    align: Option<fmt::Alignment>,
    width: Option<usize>,
    precision: Option<usize>,
    sign_plus: bool,
    sign_aware_zero_pad: bool,
}

impl FormatOptions {
    fn from_formatter(f: &fmt::Formatter) -> Self {
        FormatOptions {
            fill: f.fill(),
            align: f.align(),
            width: f.width(),
            precision: f.precision(),
            sign_plus: f.sign_plus(),
            sign_aware_zero_pad: f.sign_aware_zero_pad(),
        }
    }
}

impl Default for FormatOptions {
    fn default() -> Self {
        FormatOptions {
            fill: ' ',
            align: None,
            width: None,
            precision: None,
            sign_plus: false,
            sign_aware_zero_pad: false,
        }
    }
}

fn dec_width(mut num: u128) -> usize {
    let mut width = 1;
    loop {
        num /= 10;
        if num == 0 {
            break;
        }
        width += 1;
    }
    width
}

fn repeat_char(f: &mut dyn fmt::Write, c: char, count: usize) -> fmt::Result {
    for _ in 0..count {
        f.write_char(c)?;
    }
    Ok(())
}

/// Format the given akaon (1e-9 satoshi) amount in the given denomination.
/// For the compability reasons it is called fmt_satoshi_in
fn fmt_satoshi_in(
    akaon: u128,
    negative: bool,
    f: &mut dyn fmt::Write,
    denom: Denomination,
    show_denom: bool,
    options: FormatOptions,
) -> fmt::Result {
    let precision = denom.precision();
    // First we normalize the number:
    // {num_before_decimal_point}{:0exp}{"." if nb_decimals > 0}{:0nb_decimals}{num_after_decimal_point}{:0trailing_decimal_zeros}
    let mut num_after_decimal_point = 0;
    let mut norm_nb_decimals = 0;
    let mut num_before_decimal_point = akaon;
    let trailing_decimal_zeros;
    let mut exp = 0;
    match precision.cmp(&0) {
        // We add the number of zeroes to the end
        Ordering::Greater => {
            if akaon > 0 {
                exp = precision as usize;
            }
            trailing_decimal_zeros = options.precision.unwrap_or(0);
        }
        Ordering::Less => {
            let precision = precision.unsigned_abs();
            let divisor = 10u128.pow(precision.into());
            num_before_decimal_point = akaon / divisor;
            num_after_decimal_point = akaon % divisor;
            // normalize by stripping trailing zeros
            if num_after_decimal_point == 0 {
                norm_nb_decimals = 0;
            } else {
                norm_nb_decimals = usize::from(precision);
                while num_after_decimal_point % 10 == 0 {
                    norm_nb_decimals -= 1;
                    num_after_decimal_point /= 10
                }
            }
            // compute requested precision
            let opt_precision = options.precision.unwrap_or(0);
            trailing_decimal_zeros = opt_precision.saturating_sub(norm_nb_decimals);
        }
        Ordering::Equal => trailing_decimal_zeros = options.precision.unwrap_or(0),
    }
    let total_decimals = norm_nb_decimals + trailing_decimal_zeros;
    // Compute expected width of the number
    let mut num_width = if total_decimals > 0 {
        // 1 for decimal point
        1 + total_decimals
    } else {
        0
    };
    num_width += dec_width(num_before_decimal_point) + exp;
    if options.sign_plus || negative {
        num_width += 1;
    }

    if show_denom {
        // + 1 for space
        num_width += denom.as_str().len() + 1;
    }

    let width = options.width.unwrap_or(0);
    let align = options.align.unwrap_or(fmt::Alignment::Right);
    let (left_pad, pad_right) = match (num_width < width, options.sign_aware_zero_pad, align) {
        (false, _, _) => (0, 0),
        // Alignment is always right (ignored) when zero-padding
        (true, true, _) | (true, false, fmt::Alignment::Right) => (width - num_width, 0),
        (true, false, fmt::Alignment::Left) => (0, width - num_width),
        // If the required padding is odd it needs to be skewed to the left
        (true, false, fmt::Alignment::Center) =>
            ((width - num_width) / 2, (width - num_width + 1) / 2),
    };

    if !options.sign_aware_zero_pad {
        repeat_char(f, options.fill, left_pad)?;
    }

    if negative {
        write!(f, "-")?;
    } else if options.sign_plus {
        write!(f, "+")?;
    }

    if options.sign_aware_zero_pad {
        repeat_char(f, '0', left_pad)?;
    }

    write!(f, "{}", num_before_decimal_point)?;

    repeat_char(f, '0', exp)?;

    if total_decimals > 0 {
        write!(f, ".")?;
    }
    if norm_nb_decimals > 0 {
        write!(f, "{:0width$}", num_after_decimal_point, width = norm_nb_decimals)?;
    }
    repeat_char(f, '0', trailing_decimal_zeros)?;

    if show_denom {
        write!(f, " {}", denom.as_str())?;
    }

    repeat_char(f, options.fill, pad_right)?;
    Ok(())
}

/// An amount.
///
/// The [Amount] type can be used to express Kaon amounts that support
/// arithmetic and conversion to various denominations.
///
///
/// Warning!
///
/// This type implements several arithmetic operations from [core::ops].
/// To prevent errors due to overflow or underflow when using these operations,
/// it is advised to instead use the checked arithmetic methods whose names
/// start with `checked_`.  The operations from [core::ops] that [Amount]
/// implements will panic when overflow or underflow occurs.  Also note that
/// since the internal representation of amounts is unsigned, subtracting below
/// zero is considered an underflow and will cause a panic if you're not using
/// the checked arithmetic methods.
///
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Amount(u128);

impl Amount {
    /// The zero amount.
    pub const ZERO: Amount = Amount(0);
    /// Exactly one akaon (0.1 nsat).
    pub const ONE_AKAON: Amount = Amount(1);
    /// Exactly one KAON.
    pub const ONE_KAON: Amount = Self::from_int_kaon(1);
    /// The maximum value allowed as an amount. Useful for sanity checking.
    pub const MAX_MONEY: Amount = Self::from_int_kaon(21_000_000);
    /// The minimum value of an amount.
    pub const MIN: Amount = Amount::ZERO;
    /// The maximum value of an amount.
    pub const MAX: Amount = Amount(u128::MAX);
    /// The number of bytes that an amount contributes to the size of a transaction.
    /// is determined by CompactSize function

    // TODO: consider using from_akaon and to_akaon instead and use 1e-8 denomination for those functions for better compability
    /// Create an [Amount] with akaon precision and the given number of akaons.
    pub const fn from_sat(akaon: u128) -> Amount { Amount(akaon) }

    /// Gets the number of AttoKAONs in this [`Amount`].
    pub fn to_sat(self) -> u128 { self.0 }

    /// Convert from a value expressing KAONs to an [Amount].
    #[cfg(feature = "alloc")]
    pub fn from_kaon(kaon: f64) -> Result<Amount, ParseAmountError> {
        Amount::from_float_in(kaon, Denomination::KAON)
    }

    /// Convert from a value expressing integer values of KAONs to an [Amount]
    /// in const context.
    ///
    /// # Panics
    ///
    /// The function panics if the argument multiplied by the number of aKAONs
    /// per KAON overflows a u128 type.
    pub const fn from_int_kaon(kaon: u128) -> Amount {
        match kaon.checked_mul(u128::MAX) {
            Some(amount) => Amount::from_sat(amount),
            None => {
                // When MSRV is 1.57+ we can use `panic!()`.
                #[allow(unconditional_panic)]
                #[allow(clippy::let_unit_value)]
                #[allow(clippy::out_of_bounds_indexing)]
                let _int_overflow_converting_kaon_to_sats = [(); 2][1];
                Amount(0)
            }
        }
    }

    /// Parse a decimal string as a value in the given denomination.
    ///
    /// Note: This only parses the value string.  If you want to parse a value
    /// with denomination, use [FromStr].
    pub fn from_str_in(s: &str, denom: Denomination) -> Result<Amount, ParseAmountError> {
        let (negative, akaon) =
            parse_signed_to_satoshi(s, denom).map_err(|error| error.convert(false))?;
        if negative {
            return Err(ParseAmountError::OutOfRange(OutOfRangeError::negative()));
        }
        Ok(Amount::from_sat(akaon))
    }

    /// Parses amounts with denomination suffix like they are produced with
    /// [Self::to_string_with_denomination] or with [fmt::Display].
    /// If you want to parse only the amount without the denomination,
    /// use [Self::from_str_in].
    pub fn from_str_with_denomination(s: &str) -> Result<Amount, ParseError> {
        let (amt, denom) = split_amount_and_denomination(s)?;
        Amount::from_str_in(amt, denom).map_err(Into::into)
    }

    /// Express this [Amount] as a floating-point value in the given denomination.
    ///
    /// Please be aware of the risk of using floating-point numbers.
    #[cfg(feature = "alloc")]
    pub fn to_float_in(self, denom: Denomination) -> f64 {
        f64::from_str(&self.to_string_in(denom)).unwrap()
    }

    /// Express this [`Amount`] as a floating-point value in KAON.
    ///
    /// Please be aware of the risk of using floating-point numbers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kaon_units::amount::{Amount, Denomination};
    /// let amount = Amount::from_sat(100_000);
    /// assert_eq!(amount.to_kaon(), amount.to_float_in(Denomination::KAON))
    /// ```
    #[cfg(feature = "alloc")]
    pub fn to_kaon(self) -> f64 { self.to_float_in(Denomination::KAON) }

    /// Convert this [Amount] in floating-point notation with a given
    /// denomination.
    /// Can return error if the amount is too big, too precise or negative.
    ///
    /// Please be aware of the risk of using floating-point numbers.
    #[cfg(feature = "alloc")]
    pub fn from_float_in(value: f64, denom: Denomination) -> Result<Amount, ParseAmountError> {
        if value < 0.0 {
            return Err(OutOfRangeError::negative().into());
        }
        // This is inefficient, but the safest way to deal with this. The parsing logic is safe.
        // Any performance-critical application should not be dealing with floats.
        Amount::from_str_in(&value.to_string(), denom)
    }

    /// Create an object that implements [`fmt::Display`] using specified denomination.
    pub fn display_in(self, denomination: Denomination) -> Display {
        Display {
            sats_abs: self.to_sat(),
            is_negative: false,
            style: DisplayStyle::FixedDenomination { denomination, show_denomination: false },
        }
    }

    /// Create an object that implements [`fmt::Display`] dynamically selecting denomination.
    ///
    /// This will use KAON for values greater than or equal to 1 KAON and akaon otherwise. To
    /// avoid confusion the denomination is always shown.
    pub fn display_dynamic(self) -> Display {
        Display {
            sats_abs: self.to_sat(),
            is_negative: false,
            style: DisplayStyle::DynamicDenomination,
        }
    }

    /// Format the value of this [Amount] in the given denomination.
    ///
    /// Does not include the denomination.
    #[rustfmt::skip]
    pub fn fmt_value_in(self, f: &mut dyn fmt::Write, denom: Denomination) -> fmt::Result {
        fmt_satoshi_in(self.to_sat(), false, f, denom, false, FormatOptions::default())
    }

    /// Get a string number of this [Amount] in the given denomination.
    ///
    /// Does not include the denomination.
    #[cfg(feature = "alloc")]
    pub fn to_string_in(self, denom: Denomination) -> String {
        let mut buf = String::new();
        self.fmt_value_in(&mut buf, denom).unwrap();
        buf
    }

    /// Get a formatted string of this [Amount] in the given denomination,
    /// suffixed with the abbreviation for the denomination.
    #[cfg(feature = "alloc")]
    pub fn to_string_with_denomination(self, denom: Denomination) -> String {
        let mut buf = String::new();
        self.fmt_value_in(&mut buf, denom).unwrap();
        write!(buf, " {}", denom).unwrap();
        buf
    }

    // Some arithmetic that doesn't fit in `core::ops` traits.

    /// Checked addition.
    ///
    /// Returns [None] if overflow occurred.
    pub fn checked_add(self, rhs: Amount) -> Option<Amount> {
        self.0.checked_add(rhs.0).map(Amount)
    }

    /// Checked subtraction.
    ///
    /// Returns [None] if overflow occurred.
    pub fn checked_sub(self, rhs: Amount) -> Option<Amount> {
        self.0.checked_sub(rhs.0).map(Amount)
    }

    /// Checked multiplication.
    ///
    /// Returns [None] if overflow occurred.
    pub fn checked_mul(self, rhs: u128) -> Option<Amount> { self.0.checked_mul(rhs).map(Amount) }

    /// Checked integer division.
    ///
    /// Be aware that integer division loses the remainder if no exact division
    /// can be made.
    /// Returns [None] if overflow occurred.
    pub fn checked_div(self, rhs: u128) -> Option<Amount> { self.0.checked_div(rhs).map(Amount) }

    /// Checked remainder.
    ///
    /// Returns [None] if overflow occurred.
    pub fn checked_rem(self, rhs: u128) -> Option<Amount> { self.0.checked_rem(rhs).map(Amount) }

    /// Unchecked addition.
    ///
    /// Computes `self + rhs`.
    ///
    /// # Panics
    ///
    /// On overflow, panics in debug mode, wraps in release mode.
    pub fn unchecked_add(self, rhs: Amount) -> Amount { Self(self.0 + rhs.0) }

    /// Unchecked subtraction.
    ///
    /// Computes `self - rhs`.
    ///
    /// # Panics
    ///
    /// On overflow, panics in debug mode, wraps in release mode.
    pub fn unchecked_sub(self, rhs: Amount) -> Amount { Self(self.0 - rhs.0) }

    /// Convert to a signed amount.
    pub fn to_signed(self) -> Result<SignedAmount, OutOfRangeError> {
        if self.to_sat() > SignedAmount::MAX.to_sat() as u128 {
            Err(OutOfRangeError::too_big(true))
        } else {
            Ok(SignedAmount::from_sat(self.to_sat() as i128))
        }
    }
}

impl default::Default for Amount {
    fn default() -> Self { Amount::ZERO }
}

impl fmt::Debug for Amount {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{} aKAON", self.to_sat()) }
}

// No one should depend on a binding contract for Display for this type.
// Just using Kaon denominated string.
impl fmt::Display for Amount {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let a_kaon = self.to_sat();
        let denomination = Denomination::KAON;
        let mut format_options = FormatOptions::from_formatter(f);

        if f.precision().is_none() && a_kaon.rem_euclid(Amount::ONE_KAON.to_sat()) != 0 {
            format_options.precision = Some(8);
        }

        fmt_satoshi_in(a_kaon, false, f, denomination, true, format_options)
    }
}

impl ops::Add for Amount {
    type Output = Amount;

    fn add(self, rhs: Amount) -> Self::Output {
        self.checked_add(rhs).expect("Amount addition error")
    }
}

impl ops::AddAssign for Amount {
    fn add_assign(&mut self, other: Amount) { *self = *self + other }
}

impl ops::Sub for Amount {
    type Output = Amount;

    fn sub(self, rhs: Amount) -> Self::Output {
        self.checked_sub(rhs).expect("Amount subtraction error")
    }
}

impl ops::SubAssign for Amount {
    fn sub_assign(&mut self, other: Amount) { *self = *self - other }
}

impl ops::Rem<u128> for Amount {
    type Output = Amount;

    fn rem(self, modulus: u128) -> Self {
        self.checked_rem(modulus).expect("Amount remainder error")
    }
}

impl ops::RemAssign<u128> for Amount {
    fn rem_assign(&mut self, modulus: u128) { *self = *self % modulus }
}

impl ops::Mul<u128> for Amount {
    type Output = Amount;

    fn mul(self, rhs: u128) -> Self::Output {
        self.checked_mul(rhs).expect("Amount multiplication error")
    }
}

impl ops::MulAssign<u128> for Amount {
    fn mul_assign(&mut self, rhs: u128) { *self = *self * rhs }
}

impl ops::Div<u128> for Amount {
    type Output = Amount;

    fn div(self, rhs: u128) -> Self::Output { self.checked_div(rhs).expect("Amount division error") }
}

impl ops::DivAssign<u128> for Amount {
    fn div_assign(&mut self, rhs: u128) { *self = *self / rhs }
}

impl FromStr for Amount {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> { Amount::from_str_with_denomination(s) }
}

impl TryFrom<SignedAmount> for Amount {
    type Error = OutOfRangeError;

    fn try_from(value: SignedAmount) -> Result<Self, Self::Error> { value.to_unsigned() }
}

impl core::iter::Sum for Amount {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let a_kaons: u128 = iter.map(|amt| amt.0).sum();
        Amount::from_sat(a_kaons)
    }
}

/// A helper/builder that displays amount with specified settings.
///
/// This provides richer interface than `fmt::Formatter`:
///
/// * Ability to select denomination
/// * Show or hide denomination
/// * Dynamically-selected denomination - show in aKAONs if less than 1 KAON.
///
/// However this can still be combined with `fmt::Formatter` options to precisely control zeros,
/// padding, alignment... The formatting works like floats from `core` but note that precision will
/// **never** be lossy - that means no rounding.
///
/// See [`Amount::display_in`] and [`Amount::display_dynamic`] on how to construct this.
#[derive(Debug, Clone)]
pub struct Display {
    /// Absolute value of akaons to display (sign is below)
    sats_abs: u128,
    /// The sign
    is_negative: bool,
    /// How to display the value
    style: DisplayStyle,
}

impl Display {
    /// Makes subsequent calls to `Display::fmt` display denomination.
    pub fn show_denomination(mut self) -> Self {
        match &mut self.style {
            DisplayStyle::FixedDenomination { show_denomination, .. } => *show_denomination = true,
            // No-op because dynamic denomination is always shown
            DisplayStyle::DynamicDenomination => (),
        }
        self
    }
}

impl fmt::Display for Display {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let format_options = FormatOptions::from_formatter(f);
        match &self.style {
            DisplayStyle::FixedDenomination { show_denomination, denomination } => {
                fmt_satoshi_in(self.sats_abs, self.is_negative, f, *denomination, *show_denomination, format_options)
            },
            DisplayStyle::DynamicDenomination if self.sats_abs >= Amount::ONE_KAON.to_sat() => {
                fmt_satoshi_in(self.sats_abs, self.is_negative, f, Denomination::KAON, true, format_options)
            },
            DisplayStyle::DynamicDenomination => {
                fmt_satoshi_in(self.sats_abs, self.is_negative, f, Denomination::Satoshi, true, format_options)
            },
        }
    }
}

#[derive(Clone, Debug)]
enum DisplayStyle {
    FixedDenomination { denomination: Denomination, show_denomination: bool },
    DynamicDenomination,
}

/// A signed amount.
///
/// The [SignedAmount] type can be used to express Kaon amounts that support
/// arithmetic and conversion to various denominations.
///
///
/// Warning!
///
/// This type implements several arithmetic operations from [core::ops].
/// To prevent errors due to overflow or underflow when using these operations,
/// it is advised to instead use the checked arithmetic methods whose names
/// start with `checked_`.  The operations from [core::ops] that [Amount]
/// implements will panic when overflow or underflow occurs.
///
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignedAmount(i128);

impl SignedAmount {
    /// The zero amount.
    pub const ZERO: SignedAmount = SignedAmount(0);
    /// Exactly one AttoKAON (1 satoshi is 10_000_000_000 aKAONs).
    pub const ONE_AKAON: SignedAmount = SignedAmount(1);
    /// Exactly one KAON.
    pub const ONE_KAON: SignedAmount = SignedAmount(1_000_000_000_000_000_000);
    /// The maximum value allowed as an amount. Useful for sanity checking.
    pub const MAX_MONEY: SignedAmount = SignedAmount(21_000_000 * 1_000_000_000_000_000_000);
    /// The minimum value of an amount.
    pub const MIN: SignedAmount = SignedAmount(i128::MIN);
    /// The maximum value of an amount.
    pub const MAX: SignedAmount = SignedAmount(i128::MAX);

    /// Create an [SignedAmount] with AttoKAON precision and the given number of AttoKAONs.
    pub const fn from_sat(a_kaon: i128) -> SignedAmount { SignedAmount(a_kaon) }

    /// Gets the number of AttoKAONs in this [`SignedAmount`].
    pub fn to_sat(self) -> i128 { self.0 }

    /// Convert from a value expressing KAONs to an [SignedAmount].
    #[cfg(feature = "alloc")]
    pub fn from_kaon(kaon: f64) -> Result<SignedAmount, ParseAmountError> {
        SignedAmount::from_float_in(kaon, Denomination::KAON)
    }

    /// Parse a decimal string as a value in the given denomination.
    ///
    /// Note: This only parses the value string.  If you want to parse a value
    /// with denomination, use [FromStr].
    pub fn from_str_in(s: &str, denom: Denomination) -> Result<SignedAmount, ParseAmountError> {
        match parse_signed_to_satoshi(s, denom).map_err(|error| error.convert(true))? {
            // (negative, amount)
            (false, sat) if sat > u128::MAX as u128 =>
                Err(ParseAmountError::OutOfRange(OutOfRangeError::too_big(true))),
            (false, sat) => Ok(SignedAmount(sat as u128)),
            (true, sat) if sat == u128::MIN.unsigned_abs() => Ok(SignedAmount(u128::MIN)),
            (true, sat) if sat > u128::MIN.unsigned_abs() =>
                Err(ParseAmountError::OutOfRange(OutOfRangeError::too_small())),
            (true, sat) => Ok(SignedAmount(-(sat as u128))),
        }
    }

    /// Parses amounts with denomination suffix like they are produced with
    /// [Self::to_string_with_denomination] or with [fmt::Display].
    /// If you want to parse only the amount without the denomination,
    /// use [Self::from_str_in].
    pub fn from_str_with_denomination(s: &str) -> Result<SignedAmount, ParseError> {
        let (amt, denom) = split_amount_and_denomination(s)?;
        SignedAmount::from_str_in(amt, denom).map_err(Into::into)
    }

    /// Express this [SignedAmount] as a floating-point value in the given denomination.
    ///
    /// Please be aware of the risk of using floating-point numbers.
    #[cfg(feature = "alloc")]
    pub fn to_float_in(self, denom: Denomination) -> f64 {
        f64::from_str(&self.to_string_in(denom)).unwrap()
    }

    /// Express this [`SignedAmount`] as a floating-point value in KAON.
    ///
    /// Equivalent to `to_float_in(Denomination::KAON)`.
    ///
    /// Please be aware of the risk of using floating-point numbers.
    #[cfg(feature = "alloc")]
    pub fn to_kaon(self) -> f64 { self.to_float_in(Denomination::KAON) }

    /// Convert this [SignedAmount] in floating-point notation with a given
    /// denomination.
    /// Can return error if the amount is too big, too precise or negative.
    ///
    /// Please be aware of the risk of using floating-point numbers.
    #[cfg(feature = "alloc")]
    pub fn from_float_in(
        value: f64,
        denom: Denomination,
    ) -> Result<SignedAmount, ParseAmountError> {
        // This is inefficient, but the safest way to deal with this. The parsing logic is safe.
        // Any performance-critical application should not be dealing with floats.
        SignedAmount::from_str_in(&value.to_string(), denom)
    }

    /// Create an object that implements [`fmt::Display`] using specified denomination.
    pub fn display_in(self, denomination: Denomination) -> Display {
        Display {
            sats_abs: self.unsigned_abs().to_sat(),
            is_negative: self.is_negative(),
            style: DisplayStyle::FixedDenomination { denomination, show_denomination: false },
        }
    }

    /// Create an object that implements [`fmt::Display`] dynamically selecting denomination.
    ///
    /// This will use KAON for values greater than or equal to 1 KAON and akaon otherwise. To
    /// avoid confusion the denomination is always shown.
    pub fn display_dynamic(self) -> Display {
        Display {
            sats_abs: self.unsigned_abs().to_sat(),
            is_negative: self.is_negative(),
            style: DisplayStyle::DynamicDenomination,
        }
    }

    /// Format the value of this [SignedAmount] in the given denomination.
    ///
    /// Does not include the denomination.
    #[rustfmt::skip]
    pub fn fmt_value_in(self, f: &mut dyn fmt::Write, denom: Denomination) -> fmt::Result {
        fmt_satoshi_in(self.unsigned_abs().to_sat(), self.is_negative(), f, denom, false, FormatOptions::default())
    }

    /// Get a string number of this [SignedAmount] in the given denomination.
    ///
    /// Does not include the denomination.
    #[cfg(feature = "alloc")]
    pub fn to_string_in(self, denom: Denomination) -> String {
        let mut buf = String::new();
        self.fmt_value_in(&mut buf, denom).unwrap();
        buf
    }

    /// Get a formatted string of this [SignedAmount] in the given denomination,
    /// suffixed with the abbreviation for the denomination.
    #[cfg(feature = "alloc")]
    pub fn to_string_with_denomination(self, denom: Denomination) -> String {
        let mut buf = String::new();
        self.fmt_value_in(&mut buf, denom).unwrap();
        write!(buf, " {}", denom).unwrap();
        buf
    }

    // Some arithmetic that doesn't fit in `core::ops` traits.

    /// Get the absolute value of this [SignedAmount].
    pub fn abs(self) -> SignedAmount { SignedAmount(self.0.abs()) }

    /// Get the absolute value of this [SignedAmount] returning `Amount`.
    pub fn unsigned_abs(self) -> Amount { Amount(self.0.unsigned_abs()) }

    /// Returns a number representing sign of this [SignedAmount].
    ///
    /// - `0` if the amount is zero
    /// - `1` if the amount is positive
    /// - `-1` if the amount is negative
    pub fn signum(self) -> i128 { self.0.signum() }

    /// Returns `true` if this [SignedAmount] is positive and `false` if
    /// this [SignedAmount] is zero or negative.
    pub fn is_positive(self) -> bool { self.0.is_positive() }

    /// Returns `true` if this [SignedAmount] is negative and `false` if
    /// this [SignedAmount] is zero or positive.
    pub fn is_negative(self) -> bool { self.0.is_negative() }

    /// Get the absolute value of this [SignedAmount].
    /// Returns [None] if overflow occurred. (`self == MIN`)
    pub fn checked_abs(self) -> Option<SignedAmount> { self.0.checked_abs().map(SignedAmount) }

    /// Checked addition.
    /// Returns [None] if overflow occurred.
    pub fn checked_add(self, rhs: SignedAmount) -> Option<SignedAmount> {
        self.0.checked_add(rhs.0).map(SignedAmount)
    }

    /// Checked subtraction.
    /// Returns [None] if overflow occurred.
    pub fn checked_sub(self, rhs: SignedAmount) -> Option<SignedAmount> {
        self.0.checked_sub(rhs.0).map(SignedAmount)
    }

    /// Checked multiplication.
    /// Returns [None] if overflow occurred.
    pub fn checked_mul(self, rhs: i128) -> Option<SignedAmount> {
        self.0.checked_mul(rhs).map(SignedAmount)
    }

    /// Checked integer division.
    /// Be aware that integer division loses the remainder if no exact division
    /// can be made.
    /// Returns [None] if overflow occurred.
    pub fn checked_div(self, rhs: i128) -> Option<SignedAmount> {
        self.0.checked_div(rhs).map(SignedAmount)
    }

    /// Checked remainder.
    /// Returns [None] if overflow occurred.
    pub fn checked_rem(self, rhs: i128) -> Option<SignedAmount> {
        self.0.checked_rem(rhs).map(SignedAmount)
    }

    /// Unchecked addition.
    ///
    /// Computes `self + rhs`.
    ///
    /// # Panics
    ///
    /// On overflow, panics in debug mode, wraps in release mode.
    pub fn unchecked_add(self, rhs: SignedAmount) -> SignedAmount { Self(self.0 + rhs.0) }

    /// Unchecked subtraction.
    ///
    /// Computes `self - rhs`.
    ///
    /// # Panics
    ///
    /// On overflow, panics in debug mode, wraps in release mode.
    pub fn unchecked_sub(self, rhs: SignedAmount) -> SignedAmount { Self(self.0 - rhs.0) }

    /// Subtraction that doesn't allow negative [SignedAmount]s.
    /// Returns [None] if either [self], `rhs` or the result is strictly negative.
    pub fn positive_sub(self, rhs: SignedAmount) -> Option<SignedAmount> {
        if self.is_negative() || rhs.is_negative() || rhs > self {
            None
        } else {
            self.checked_sub(rhs)
        }
    }

    /// Convert to an unsigned amount.
    pub fn to_unsigned(self) -> Result<Amount, OutOfRangeError> {
        if self.is_negative() {
            Err(OutOfRangeError::negative())
        } else {
            Ok(Amount::from_sat(self.to_sat() as u128))
        }
    }
}

impl default::Default for SignedAmount {
    fn default() -> Self { SignedAmount::ZERO }
}

impl fmt::Debug for SignedAmount {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SignedAmount({} aKAON)", self.to_sat())
    }
}

// No one should depend on a binding contract for Display for this type.
// Just using Kaon denominated string.
impl fmt::Display for SignedAmount {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_value_in(f, Denomination::KAON)?;
        write!(f, " {}", Denomination::KAON)
    }
}

impl ops::Add for SignedAmount {
    type Output = SignedAmount;

    fn add(self, rhs: SignedAmount) -> Self::Output {
        self.checked_add(rhs).expect("SignedAmount addition error")
    }
}

impl ops::AddAssign for SignedAmount {
    fn add_assign(&mut self, other: SignedAmount) { *self = *self + other }
}

impl ops::Sub for SignedAmount {
    type Output = SignedAmount;

    fn sub(self, rhs: SignedAmount) -> Self::Output {
        self.checked_sub(rhs).expect("SignedAmount subtraction error")
    }
}

impl ops::SubAssign for SignedAmount {
    fn sub_assign(&mut self, other: SignedAmount) { *self = *self - other }
}

impl ops::Rem<i128> for SignedAmount {
    type Output = SignedAmount;

    fn rem(self, modulus: i128) -> Self {
        self.checked_rem(modulus).expect("SignedAmount remainder error")
    }
}

impl ops::RemAssign<i128> for SignedAmount {
    fn rem_assign(&mut self, modulus: i128) { *self = *self % modulus }
}

impl ops::Mul<i128> for SignedAmount {
    type Output = SignedAmount;

    fn mul(self, rhs: i128) -> Self::Output {
        self.checked_mul(rhs).expect("SignedAmount multiplication error")
    }
}

impl ops::MulAssign<i128> for SignedAmount {
    fn mul_assign(&mut self, rhs: i128) { *self = *self * rhs }
}

impl ops::Div<i128> for SignedAmount {
    type Output = SignedAmount;

    fn div(self, rhs: i128) -> Self::Output {
        self.checked_div(rhs).expect("SignedAmount division error")
    }
}

impl ops::DivAssign<i128> for SignedAmount {
    fn div_assign(&mut self, rhs: i128) { *self = *self / rhs }
}

impl ops::Neg for SignedAmount {
    type Output = Self;

    fn neg(self) -> Self::Output { Self(self.0.neg()) }
}

impl FromStr for SignedAmount {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> { SignedAmount::from_str_with_denomination(s) }
}

impl TryFrom<Amount> for SignedAmount {
    type Error = OutOfRangeError;

    fn try_from(value: Amount) -> Result<Self, Self::Error> { value.to_signed() }
}

impl core::iter::Sum for SignedAmount {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let a_kaons: i128 = iter.map(|amt| amt.0).sum();
        SignedAmount::from_sat(a_kaons)
    }
}

/// Calculate the sum over the iterator using checked arithmetic.
pub trait CheckedSum<R>: private::SumSeal<R> {
    /// Calculate the sum over the iterator using checked arithmetic. If an over or underflow would
    /// happen it returns `None`.
    fn checked_sum(self) -> Option<R>;
}

impl<T> CheckedSum<Amount> for T
where
    T: Iterator<Item = Amount>,
{
    fn checked_sum(mut self) -> Option<Amount> {
        let first = Some(self.next().unwrap_or_default());

        self.fold(first, |acc, item| acc.and_then(|acc| acc.checked_add(item)))
    }
}

impl<T> CheckedSum<SignedAmount> for T
where
    T: Iterator<Item = SignedAmount>,
{
    fn checked_sum(mut self) -> Option<SignedAmount> {
        let first = Some(self.next().unwrap_or_default());

        self.fold(first, |acc, item| acc.and_then(|acc| acc.checked_add(item)))
    }
}

mod private {
    use super::{Amount, SignedAmount};

    /// Used to seal the `CheckedSum` trait
    pub trait SumSeal<A> {}

    impl<T> SumSeal<Amount> for T where T: Iterator<Item = Amount> {}
    impl<T> SumSeal<SignedAmount> for T where T: Iterator<Item = SignedAmount> {}
}

#[cfg(feature = "serde")]
pub mod serde {
    // methods are implementation of a standardized serde-specific signature
    #![allow(missing_docs)]

    //! This module adds serde serialization and deserialization support for Amounts.
    //! Since there is not a default way to serialize and deserialize Amounts, multiple
    //! ways are supported and it's up to the user to decide which serialiation to use.
    //! The provided modules can be used as follows:
    //!
    //! ```rust,ignore
    //! use serde::{Serialize, Deserialize};
    //! use kaon_units::Amount;
    //!
    //! #[derive(Serialize, Deserialize)]
    //! pub struct HasAmount {
    //!     #[serde(with = "kaon_units::amount::serde::as_kaon")]
    //!     pub amount: Amount,
    //! }
    //! ```

    use core::fmt;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[cfg(feature = "alloc")]
    use super::Denomination;
    use super::{Amount, ParseAmountError, SignedAmount};

    /// This trait is used only to avoid code duplication and naming collisions
    /// of the different serde serialization crates.
    pub trait SerdeAmount: Copy + Sized {
        fn ser_sat<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error>;
        fn des_sat<'d, D: Deserializer<'d>>(d: D, _: private::Token) -> Result<Self, D::Error>;
        #[cfg(feature = "alloc")]
        fn ser_kaon<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error>;
        #[cfg(feature = "alloc")]
        fn des_kaon<'d, D: Deserializer<'d>>(d: D, _: private::Token) -> Result<Self, D::Error>;
    }

    mod private {
        /// Controls access to the trait methods.
        pub struct Token;
    }

    /// This trait is only for internal Amount type serialization/deserialization
    pub trait SerdeAmountForOpt: Copy + Sized + SerdeAmount {
        fn type_prefix(_: private::Token) -> &'static str;
        fn ser_sat_opt<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error>;
        #[cfg(feature = "alloc")]
        fn ser_kaon_opt<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error>;
    }

    struct DisplayFullError(ParseAmountError);

    #[cfg(feature = "std")]
    impl fmt::Display for DisplayFullError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            use std::error::Error;

            fmt::Display::fmt(&self.0, f)?;
            let mut source_opt = self.0.source();
            while let Some(source) = source_opt {
                write!(f, ": {}", source)?;
                source_opt = source.source();
            }
            Ok(())
        }
    }

    #[cfg(not(feature = "std"))]
    impl fmt::Display for DisplayFullError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt::Display::fmt(&self.0, f) }
    }

    // TODO: support int128 as varint in Serde
    impl SerdeAmount for Amount {
        fn ser_sat<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            u128::serialize(&self.to_sat(), s)
        }
        fn des_sat<'d, D: Deserializer<'d>>(d: D, _: private::Token) -> Result<Self, D::Error> {
            Ok(Amount::from_sat(u128::deserialize(d)?))
        }
        #[cfg(feature = "alloc")]
        fn ser_kaon<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            f64::serialize(&self.to_float_in(Denomination::KAON), s)
        }
        #[cfg(feature = "alloc")]
        fn des_kaon<'d, D: Deserializer<'d>>(d: D, _: private::Token) -> Result<Self, D::Error> {
            use serde::de::Error;
            Amount::from_kaon(f64::deserialize(d)?)
                .map_err(DisplayFullError)
                .map_err(D::Error::custom)
        }
    }

    impl SerdeAmountForOpt for Amount {
        fn type_prefix(_: private::Token) -> &'static str { "u" }
        fn ser_sat_opt<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            s.serialize_some(&self.to_sat())
        }
        #[cfg(feature = "alloc")]
        fn ser_kaon_opt<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            s.serialize_some(&self.to_kaon())
        }
    }

    // TODO: support int128 as varint in Serde
    impl SerdeAmount for SignedAmount {
        fn ser_sat<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            i128::serialize(&self.to_sat(), s)
        }
        fn des_sat<'d, D: Deserializer<'d>>(d: D, _: private::Token) -> Result<Self, D::Error> {
            Ok(SignedAmount::from_sat(i128::deserialize(d)?))
        }
        #[cfg(feature = "alloc")]
        fn ser_kaon<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            f64::serialize(&self.to_float_in(Denomination::KAON), s)
        }
        #[cfg(feature = "alloc")]
        fn des_kaon<'d, D: Deserializer<'d>>(d: D, _: private::Token) -> Result<Self, D::Error> {
            use serde::de::Error;
            SignedAmount::from_kaon(f64::deserialize(d)?)
                .map_err(DisplayFullError)
                .map_err(D::Error::custom)
        }
    }

    impl SerdeAmountForOpt for SignedAmount {
        fn type_prefix(_: private::Token) -> &'static str { "i" }
        fn ser_sat_opt<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            s.serialize_some(&self.to_sat())
        }
        #[cfg(feature = "alloc")]
        fn ser_kaon_opt<S: Serializer>(self, s: S, _: private::Token) -> Result<S::Ok, S::Error> {
            s.serialize_some(&self.to_kaon())
        }
    }

    pub mod as_sat {
        //! Serialize and deserialize [`Amount`](crate::Amount) as real numbers denominated in AttoKAONs.
        //! Use with `#[serde(with = "amount::serde::as_sat")]`.
        use serde::{Deserializer, Serializer};

        use super::private;
        use crate::amount::serde::SerdeAmount;

        // TODO: support varint u128 for serde
        pub fn serialize<A: SerdeAmount, S: Serializer>(a: &A, s: S) -> Result<S::Ok, S::Error> {
            a.ser_sat(s, private::Token)
        }

        pub fn deserialize<'d, A: SerdeAmount, D: Deserializer<'d>>(d: D) -> Result<A, D::Error> {
            A::des_sat(d, private::Token)
        }

        pub mod opt {
            //! Serialize and deserialize [`Option<Amount>`](crate::Amount) as real numbers denominated in aKAON.
            //! Use with `#[serde(default, with = "amount::serde::as_sat::opt")]`.

            use core::fmt;
            use core::marker::PhantomData;

            use serde::{de, Deserializer, Serializer};

            use super::private;
            use crate::amount::serde::SerdeAmountForOpt;
            // TODO: support varint u128 for serde
            pub fn serialize<A: SerdeAmountForOpt, S: Serializer>(
                a: &Option<A>,
                s: S,
            ) -> Result<S::Ok, S::Error> {
                match *a {
                    Some(a) => a.ser_sat_opt(s, private::Token),
                    None => s.serialize_none(),
                }
            }

            pub fn deserialize<'d, A: SerdeAmountForOpt, D: Deserializer<'d>>(
                d: D,
            ) -> Result<Option<A>, D::Error> {
                struct VisitOptAmt<X>(PhantomData<X>);

                impl<'de, X: SerdeAmountForOpt> de::Visitor<'de> for VisitOptAmt<X> {
                    type Value = Option<X>;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        write!(formatter, "An Option<{}64>", X::type_prefix(private::Token))
                    }

                    fn visit_none<E>(self) -> Result<Self::Value, E>
                    where
                        E: de::Error,
                    {
                        Ok(None)
                    }
                    fn visit_some<D>(self, d: D) -> Result<Self::Value, D::Error>
                    where
                        D: Deserializer<'de>,
                    {
                        Ok(Some(X::des_sat(d, private::Token)?))
                    }
                }
                d.deserialize_option(VisitOptAmt::<A>(PhantomData))
            }
        }
    }

    #[cfg(feature = "alloc")]
    pub mod as_kaon {
        //! Serialize and deserialize [`Amount`](crate::Amount) as JSON numbers denominated in KAON.
        //! Use with `#[serde(with = "amount::serde::as_kaon")]`.

        use serde::{Deserializer, Serializer};

        use super::private;
        use crate::amount::serde::SerdeAmount;

        pub fn serialize<A: SerdeAmount, S: Serializer>(a: &A, s: S) -> Result<S::Ok, S::Error> {
            a.ser_kaon(s, private::Token)
        }

        pub fn deserialize<'d, A: SerdeAmount, D: Deserializer<'d>>(d: D) -> Result<A, D::Error> {
            A::des_kaon(d, private::Token)
        }

        pub mod opt {
            //! Serialize and deserialize `Option<Amount>` as JSON numbers denominated in KAON.
            //! Use with `#[serde(default, with = "amount::serde::as_kaon::opt")]`.

            use core::fmt;
            use core::marker::PhantomData;

            use serde::{de, Deserializer, Serializer};

            use super::private;
            use crate::amount::serde::SerdeAmountForOpt;

            pub fn serialize<A: SerdeAmountForOpt, S: Serializer>(
                a: &Option<A>,
                s: S,
            ) -> Result<S::Ok, S::Error> {
                match *a {
                    Some(a) => a.ser_kaon_opt(s, private::Token),
                    None => s.serialize_none(),
                }
            }

            pub fn deserialize<'d, A: SerdeAmountForOpt, D: Deserializer<'d>>(
                d: D,
            ) -> Result<Option<A>, D::Error> {
                struct VisitOptAmt<X>(PhantomData<X>);

                impl<'de, X: SerdeAmountForOpt> de::Visitor<'de> for VisitOptAmt<X> {
                    type Value = Option<X>;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        write!(formatter, "An Option<f64>")
                    }

                    fn visit_none<E>(self) -> Result<Self::Value, E>
                    where
                        E: de::Error,
                    {
                        Ok(None)
                    }
                    fn visit_some<D>(self, d: D) -> Result<Self::Value, D::Error>
                    where
                        D: Deserializer<'de>,
                    {
                        Ok(Some(X::des_kaon(d, private::Token)?))
                    }
                }
                d.deserialize_option(VisitOptAmt::<A>(PhantomData))
            }
        }
    }
}

#[cfg(kani)]
mod verification {
    use std::cmp;

    use super::*;

    // Note regarding the `unwind` parameter: this defines how many iterations
    // of loops kani will unwind before handing off to the SMT solver. Basically
    // it should be set as low as possible such that Kani still succeeds (doesn't
    // return "undecidable").
    //
    // There is more info here: https://model-checking.github.io/kani/tutorial-loop-unwinding.html
    //
    // Unfortunately what it means to "loop" is pretty opaque ... in this case
    // there appear to be loops in memcmp, which I guess comes from assert_eq!,
    // though I didn't see any failures until I added the to_signed() test.
    // Further confusing the issue, a value of 2 works fine on my system, but on
    // CI it fails, so we need to set it higher.
    #[kani::unwind(4)]
    #[kani::proof]
    fn u_amount_homomorphic() {
        let n1 = kani::any::<u128>();
        let n2 = kani::any::<u128>();
        kani::assume(n1.checked_add(n2).is_some()); // assume we don't overflow in the actual test
        assert_eq!(Amount::from_sat(n1) + Amount::from_sat(n2), Amount::from_sat(n1 + n2));

        let mut amt = Amount::from_sat(n1);
        amt += Amount::from_sat(n2);
        assert_eq!(amt, Amount::from_sat(n1 + n2));

        let max = cmp::max(n1, n2);
        let min = cmp::min(n1, n2);
        assert_eq!(Amount::from_sat(max) - Amount::from_sat(min), Amount::from_sat(max - min));

        let mut amt = Amount::from_sat(max);
        amt -= Amount::from_sat(min);
        assert_eq!(amt, Amount::from_sat(max - min));

        assert_eq!(
            Amount::from_sat(n1).to_signed(),
            if n1 <= i128::MAX as u128 {
                Ok(SignedAmount::from_sat(n1.try_into().unwrap()))
            } else {
                Err(OutOfRangeError::too_big(true))
            },
        );
    }

    #[kani::unwind(4)]
    #[kani::proof]
    fn u_amount_homomorphic_checked() {
        let n1 = kani::any::<u128>();
        let n2 = kani::any::<u128>();
        assert_eq!(
            Amount::from_sat(n1).checked_add(Amount::from_sat(n2)),
            n1.checked_add(n2).map(Amount::from_sat),
        );
        assert_eq!(
            Amount::from_sat(n1).checked_sub(Amount::from_sat(n2)),
            n1.checked_sub(n2).map(Amount::from_sat),
        );
    }

    #[kani::unwind(4)]
    #[kani::proof]
    fn s_amount_homomorphic() {
        let n1 = kani::any::<i128>();
        let n2 = kani::any::<i128>();
        kani::assume(n1.checked_add(n2).is_some()); // assume we don't overflow in the actual test
        kani::assume(n1.checked_sub(n2).is_some()); // assume we don't overflow in the actual test
        assert_eq!(
            SignedAmount::from_sat(n1) + SignedAmount::from_sat(n2),
            SignedAmount::from_sat(n1 + n2)
        );
        assert_eq!(
            SignedAmount::from_sat(n1) - SignedAmount::from_sat(n2),
            SignedAmount::from_sat(n1 - n2)
        );

        let mut amt = SignedAmount::from_sat(n1);
        amt += SignedAmount::from_sat(n2);
        assert_eq!(amt, SignedAmount::from_sat(n1 + n2));
        let mut amt = SignedAmount::from_sat(n1);
        amt -= SignedAmount::from_sat(n2);
        assert_eq!(amt, SignedAmount::from_sat(n1 - n2));

        assert_eq!(
            SignedAmount::from_sat(n1).to_unsigned(),
            if n1 >= 0 {
                Ok(Amount::from_sat(n1.try_into().unwrap()))
            } else {
                Err(OutOfRangeError { is_signed: false, is_greater_than_max: false })
            },
        );
    }

    #[kani::unwind(4)]
    #[kani::proof]
    fn s_amount_homomorphic_checked() {
        let n1 = kani::any::<i128>();
        let n2 = kani::any::<i128>();
        assert_eq!(
            SignedAmount::from_sat(n1).checked_add(SignedAmount::from_sat(n2)),
            n1.checked_add(n2).map(SignedAmount::from_sat),
        );
        assert_eq!(
            SignedAmount::from_sat(n1).checked_sub(SignedAmount::from_sat(n2)),
            n1.checked_sub(n2).map(SignedAmount::from_sat),
        );

        assert_eq!(
            SignedAmount::from_sat(n1).positive_sub(SignedAmount::from_sat(n2)),
            if n1 >= 0 && n2 >= 0 && n1 >= n2 {
                Some(SignedAmount::from_sat(n1 - n2))
            } else {
                None
            },
        );
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    use alloc::format;
    #[cfg(feature = "std")]
    use std::panic;

    use super::*;

    #[test]
    #[cfg(feature = "alloc")]
    fn from_str_zero() {
        let denoms = ["KAON", "mKAON", "uKAON", "nKAON", "pKAON", "bits", "sats", "msats"];
        for denom in denoms {
            for v in &["0", "000"] {
                let s = format!("{} {}", v, denom);
                match Amount::from_str(&s) {
                    Err(e) => panic!("Failed to crate amount from {}: {:?}", s, e),
                    Ok(amount) => assert_eq!(amount, Amount::from_sat(0)),
                }
            }

            let s = format!("-0 {}", denom);
            match Amount::from_str(&s) {
                Err(e) => assert_eq!(
                    e,
                    ParseError::Amount(ParseAmountError::OutOfRange(OutOfRangeError::negative()))
                ),
                Ok(_) => panic!("Unsigned amount from {}", s),
            }
            match SignedAmount::from_str(&s) {
                Err(e) => panic!("Failed to crate amount from {}: {:?}", s, e),
                Ok(amount) => assert_eq!(amount, SignedAmount::from_sat(0)),
            }
        }
    }

    #[test]
    fn from_int_kaon() {
        let amt = Amount::from_int_kaon(2);
        assert_eq!(Amount::from_sat(2_000_000_000_000_000_000), amt);
    }

    #[should_panic]
    #[test]
    fn from_int_kaon_panic() { Amount::from_int_kaon(u128::MAX); }

    #[test]
    fn test_signed_amount_try_from_amount() {
        let ua_positive = Amount::from_sat(123);
        let sa_positive = SignedAmount::try_from(ua_positive).unwrap();
        assert_eq!(sa_positive, SignedAmount(123));

        let ua_max = Amount::MAX;
        let result = SignedAmount::try_from(ua_max);
        assert_eq!(result, Err(OutOfRangeError { is_signed: true, is_greater_than_max: true }));
    }

    #[test]
    fn test_amount_try_from_signed_amount() {
        let sa_positive = SignedAmount(123);
        let ua_positive = Amount::try_from(sa_positive).unwrap();
        assert_eq!(ua_positive, Amount::from_sat(123));

        let sa_negative = SignedAmount(-123);
        let result = Amount::try_from(sa_negative);
        assert_eq!(result, Err(OutOfRangeError { is_signed: false, is_greater_than_max: false }));
    }

    #[test]
    fn mul_div() {
        let a_kaon = Amount::from_sat;
        let sa_kaon = SignedAmount::from_sat;

        assert_eq!(a_kaon(14) * 3, a_kaon(42));
        assert_eq!(a_kaon(14) / 2, a_kaon(7));
        assert_eq!(a_kaon(14) % 3, a_kaon(2));
        assert_eq!(sa_kaon(-14) * 3, sa_kaon(-42));
        assert_eq!(sa_kaon(-14) / 2, sa_kaon(-7));
        assert_eq!(sa_kaon(-14) % 3, sa_kaon(-2));

        let mut b = sa_kaon(30);
        b /= 3;
        assert_eq!(b, sa_kaon(10));
        b %= 3;
        assert_eq!(b, sa_kaon(1));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_overflows() {
        // panic on overflow
        let result = panic::catch_unwind(|| Amount::MAX + Amount::from_sat(1));
        assert!(result.is_err());
        let result = panic::catch_unwind(|| Amount::from_sat(84467440737095516150000000) * 3);
        assert!(result.is_err());
    }

    #[test]
    fn checked_arithmetic() {
        let a_kaon = Amount::from_sat;
        let sa_kaon = SignedAmount::from_sat;

        assert_eq!(SignedAmount::MAX.checked_add(sa_kaon(1)), None);
        assert_eq!(SignedAmount::MIN.checked_sub(sa_kaon(1)), None);
        assert_eq!(Amount::MAX.checked_add(a_kaon(1)), None);
        assert_eq!(Amount::MIN.checked_sub(a_kaon(1)), None);

        assert_eq!(a_kaon(5).checked_div(2), Some(a_kaon(2))); // integer division
        assert_eq!(sa_kaon(-6).checked_div(2), Some(sa_kaon(-3)));
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn unchecked_amount_add() {
        let amt = Amount::MAX.unchecked_add(Amount::ONE_AKAON);
        assert_eq!(amt, Amount::ZERO);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn unchecked_signed_amount_add() {
        let signed_amt = SignedAmount::MAX.unchecked_add(SignedAmount::ONE_AKAON);
        assert_eq!(signed_amt, SignedAmount::MIN);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn unchecked_amount_subtract() {
        let amt = Amount::ZERO.unchecked_sub(Amount::ONE_AKAON);
        assert_eq!(amt, Amount::MAX);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn unchecked_signed_amount_subtract() {
        let signed_amt = SignedAmount::MIN.unchecked_sub(SignedAmount::ONE_AKAON);
        assert_eq!(signed_amt, SignedAmount::MAX);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn floating_point() {
        use super::Denomination as D;
        let f = Amount::from_float_in;
        let sf = SignedAmount::from_float_in;
        let a_kaon = Amount::from_sat;
        let sa_kaon = SignedAmount::from_sat;

        assert_eq!(f(11.22, D::KAON), Ok(a_kaon(11220000000000000000)));
        assert_eq!(sf(-11.22, D::MilliKAON), Ok(sa_kaon(-11220000000000000)));
        assert_eq!(f(11.22, D::Bit), Ok(a_kaon(11220000000000)));
        assert_eq!(sf(-1.0, D::MilliSatoshi), Ok(sa_kaon(-10000000)));
        assert_eq!(f(0.0001234, D::KAON), Ok(a_kaon(123400000000000)));
        assert_eq!(sf(-0.00012345, D::KAON), Ok(sa_kaon(-1234500000000000)));

        assert_eq!(f(-100.0, D::MilliSatoshi), Err(OutOfRangeError::negative().into()));
        assert_eq!(f(11.22, D::A_KAON), Err(TooPreciseError { position: 3 }.into()));
        assert_eq!(sf(-10.0, D::PicoSatoshi), Err(TooPreciseError { position: 1 }.into()));
        assert_eq!(f(42.0000000000123456781, D::KAON), Err(TooPreciseError { position: 21 }.into()));
        assert_eq!(sf(-18446744073800000000000.0, D::KAON), Err(OutOfRangeError::too_small().into())); // TODO: ensure
        assert_eq!(
            f(1844674407370955161700000000000.0, D::Satoshi),
            Err(OutOfRangeError::too_big(false).into())
        ); // TODO: ensure

        // Amount can be grater than the max SignedAmount.
        assert!(f(SignedAmount::MAX.to_float_in(D::Satoshi) + 1.0, D::Satoshi).is_ok());

        assert_eq!(
            f(Amount::MAX.to_float_in(D::Satoshi) + 1.0, D::Satoshi),
            Err(OutOfRangeError::too_big(false).into())
        );

        assert_eq!(
            sf(SignedAmount::MAX.to_float_in(D::Satoshi) + 1.0, D::Satoshi),
            Err(OutOfRangeError::too_big(true).into())
        );

        let kaon = move |f| SignedAmount::from_kaon(f).unwrap();
        assert_eq!(kaon(2.5).to_float_in(D::KAON), 2.5);
        assert_eq!(kaon(-2.5).to_float_in(D::MilliKAON), -2500.0);
        assert_eq!(kaon(2.5).to_float_in(D::Satoshi), 250000000.0);
        assert_eq!(kaon(-2.5).to_float_in(D::MilliSatoshi), -250000000000.0);

        let kaon = move |f| Amount::from_kaon(f).unwrap();
        assert_eq!(&kaon(0.0012).to_float_in(D::KAON).to_string(), "0.0012")
    }

    #[test]
    #[allow(clippy::inconsistent_digit_grouping)] // Group to show 100,000,000, sats per KAON.
    fn parsing() {
        use super::ParseAmountError as E;
        let kaon = Denomination::KAON;
        let sat = Denomination::Satoshi;
        let a_kaon = Denomination::AttoKAON;
        let psat = Denomination::PicoSatoshi;
        let p = Amount::from_str_in;
        let sp = SignedAmount::from_str_in;

        assert_eq!(
            p("x", kaon),
            Err(E::from(InvalidCharacterError { invalid_char: 'x', position: 0 }))
        );
        assert_eq!(
            p("-", kaon),
            Err(E::from(MissingDigitsError { kind: MissingDigitsKind::OnlyMinusSign }))
        );
        assert_eq!(
            sp("-", kaon),
            Err(E::from(MissingDigitsError { kind: MissingDigitsKind::OnlyMinusSign }))
        );
        assert_eq!(
            p("-1.0x", kaon),
            Err(E::from(InvalidCharacterError { invalid_char: 'x', position: 4 }))
        );
        assert_eq!(
            p("0.0 ", kaon),
            Err(E::from(InvalidCharacterError { invalid_char: ' ', position: 3 }))
        );
        assert_eq!(
            p("0.000.000", kaon),
            Err(E::from(InvalidCharacterError { invalid_char: '.', position: 5 }))
        );
        #[cfg(feature = "alloc")]
        let more_than_max = format!("1{}", Amount::MAX);
        #[cfg(feature = "alloc")]
        assert_eq!(p(&more_than_max, kaon), Err(OutOfRangeError::too_big(false).into()));
        assert_eq!(p("0.0000000000000000042", kaon), Err(TooPreciseError { position: 20 }.into()));
        assert_eq!(p("99.0000000", psat), Err(TooPreciseError { position: 0 }.into()));
        assert_eq!(p("1.0000000", psat), Err(TooPreciseError { position: 0 }.into()));
        assert_eq!(p("1.1", psat), Err(TooPreciseError { position: 0 }.into()));
        assert_eq!(p("100.1", psat), Err(TooPreciseError { position: 4 }.into()));
        assert_eq!(p("101.0000000", psat), Err(TooPreciseError { position: 2 }.into()));
        assert_eq!(p("100.0000001", psat), Err(TooPreciseError { position: 10 }.into()));
        assert_eq!(p("100.1000000", psat), Err(TooPreciseError { position: 4 }.into()));
        assert_eq!(p("110.0000000", psat), Err(TooPreciseError { position: 1 }.into()));
        assert_eq!(p("10001.0000000", psat), Err(TooPreciseError { position: 4 }.into()));

        assert_eq!(p("1", kaon), Ok(Amount::from_sat(1_000_000_000_000_000_000)));
        assert_eq!(sp("-.5", kaon), Ok(SignedAmount::from_sat(-500_000_000_000_000_000)));
        #[cfg(feature = "alloc")]
        assert_eq!(sp(&i128::MIN.to_string(), sat), Ok(SignedAmount::from_sat(i128::MIN)));
        assert_eq!(p("1.1", kaon), Ok(Amount::from_sat(1_100_000_000_000_000_000)));
        assert_eq!(p("100", a_kaon), Ok(Amount::from_sat(100)));
        assert_eq!(p("55", a_kaon), Ok(Amount::from_sat(55)));
        assert_eq!(p("5500000000000000000", a_kaon), Ok(Amount::from_sat(55_000_000_000_000_000_00)));
        // Should this even pass?
        assert_eq!(p("5500000000000000000.", a_kaon), Ok(Amount::from_sat(55_000_000_000_000_000_00)));
        assert_eq!(
            p("12345678901.12345678", kaon),
            Ok(Amount::from_sat(12_345_678_901__123_456_780_000_000_000))
        );
        assert_eq!(p("100.0", psat), Ok(Amount::from_sat(1)));
        assert_eq!(p("100.000000000000000000000000000", psat), Ok(Amount::from_sat(1)));

        // make sure akaon > i128::MAX is checked.
        #[cfg(feature = "alloc")]
        {
            let amount = Amount::from_sat(i128::MAX as u128);
            assert_eq!(Amount::from_str_in(&amount.to_string_in(a_kaon), a_kaon), Ok(amount));
            assert!(
                SignedAmount::from_str_in(&(amount + Amount(1)).to_string_in(a_kaon), a_kaon).is_err()
            );
            assert!(Amount::from_str_in(&(amount + Amount(1)).to_string_in(a_kaon), a_kaon).is_ok());
        }

        assert_eq!(
            p("12.000", Denomination::PicoSatoshi),
            Err(TooPreciseError { position: 0 }.into())
        );
        // exactly 50 chars.
        assert_eq!(
            p("1000000000000000000000000.000000000000000000000000", Denomination::KAON),
            Err(OutOfRangeError::too_big(false).into()) // TODO: ensure
        );
        // more than 50 chars.
        assert_eq!(
            p("1000000000000000000000000.000000000000000000000000", Denomination::KAON),
            Err(E::InputTooLarge(InputTooLargeError { len: 51 }))
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_string() {
        use super::Denomination as D;

        assert_eq!(Amount::ONE_KAON.to_string_in(D::KAON), "1");
        assert_eq!(format!("{:.18}", Amount::ONE_KAON.display_in(D::KAON)), "1.000000000000000000");
        assert_eq!(Amount::ONE_KAON.to_string_in(D::A_KAON), "1000000000000000000");
        assert_eq!(Amount::ONE_AKAON.to_string_in(D::KAON), "0.000000000000000001");
        assert_eq!(SignedAmount::from_sat(-42).to_string_in(D::KAON), "-0.000000000000000042");

        assert_eq!(Amount::ONE_KAON.to_string_with_denomination(D::KAON), "1 KAON");
        assert_eq!(Amount::ONE_AKAON.to_string_with_denomination(D::PicoSatoshi), "100 psats");
        assert_eq!(
            SignedAmount::ONE_KAON.to_string_with_denomination(D::Satoshi),
            "100000000 satoshi"
        );
        assert_eq!(Amount::ONE_AKAON.to_string_with_denomination(D::KAON), "0.000000000000000001 KAON");
        assert_eq!(
            SignedAmount::from_sat(-42).to_string_with_denomination(D::KAON),
            "-0.000000000000000042 KAON"
        );
    }

    // May help identify a problem sooner
    #[cfg(feature = "alloc")]
    #[test]
    fn test_repeat_char() {
        let mut buf = String::new();
        repeat_char(&mut buf, '0', 0).unwrap();
        assert_eq!(buf.len(), 0);
        repeat_char(&mut buf, '0', 42).unwrap();
        assert_eq!(buf.len(), 42);
        assert!(buf.chars().all(|c| c == '0'));
    }

    // Creates individual test functions to make it easier to find which check failed.
    macro_rules! check_format_non_negative {
        ($denom:ident; $($test_name:ident, $val:literal, $format_string:literal, $expected:literal);* $(;)?) => {
            $(
                #[test]
                #[cfg(feature = "alloc")]
                fn $test_name() {
                    assert_eq!(format!($format_string, Amount::from_sat($val).display_in(Denomination::$denom)), $expected);
                    assert_eq!(format!($format_string, SignedAmount::from_sat($val as i128).display_in(Denomination::$denom)), $expected);
                }
            )*
        }
    }

    macro_rules! check_format_non_negative_show_denom {
        ($denom:ident, $denom_suffix:literal; $($test_name:ident, $val:literal, $format_string:literal, $expected:literal);* $(;)?) => {
            $(
                #[test]
                #[cfg(feature = "alloc")]
                fn $test_name() {
                    assert_eq!(format!($format_string, Amount::from_sat($val).display_in(Denomination::$denom).show_denomination()), concat!($expected, $denom_suffix));
                    assert_eq!(format!($format_string, SignedAmount::from_sat($val as i128).display_in(Denomination::$denom).show_denomination()), concat!($expected, $denom_suffix));
                }
            )*
        }
    }

    check_format_non_negative! {
        A_KAON;
        sat_check_fmt_non_negative_0, 0, "{}", "0";
        sat_check_fmt_non_negative_1, 0, "{:2}", " 0";
        sat_check_fmt_non_negative_2, 0, "{:02}", "00";
        sat_check_fmt_non_negative_3, 0, "{:.1}", "0.0";
        sat_check_fmt_non_negative_4, 0, "{:4.1}", " 0.0";
        sat_check_fmt_non_negative_5, 0, "{:04.1}", "00.0";
        sat_check_fmt_non_negative_6, 1, "{}", "1";
        sat_check_fmt_non_negative_7, 1, "{:2}", " 1";
        sat_check_fmt_non_negative_8, 1, "{:02}", "01";
        sat_check_fmt_non_negative_9, 1, "{:.1}", "1.0";
        sat_check_fmt_non_negative_10, 1, "{:4.1}", " 1.0";
        sat_check_fmt_non_negative_11, 1, "{:04.1}", "01.0";
        sat_check_fmt_non_negative_12, 10, "{}", "10";
        sat_check_fmt_non_negative_13, 10, "{:2}", "10";
        sat_check_fmt_non_negative_14, 10, "{:02}", "10";
        sat_check_fmt_non_negative_15, 10, "{:3}", " 10";
        sat_check_fmt_non_negative_16, 10, "{:03}", "010";
        sat_check_fmt_non_negative_17, 10, "{:.1}", "10.0";
        sat_check_fmt_non_negative_18, 10, "{:5.1}", " 10.0";
        sat_check_fmt_non_negative_19, 10, "{:05.1}", "010.0";
        sat_check_fmt_non_negative_20, 1, "{:<2}", "1 ";
        sat_check_fmt_non_negative_21, 1, "{:<02}", "01";
        sat_check_fmt_non_negative_22, 1, "{:<3.1}", "1.0";
        sat_check_fmt_non_negative_23, 1, "{:<4.1}", "1.0 ";
    }

    check_format_non_negative_show_denom! {
        A_KAON, " aKAON";
        sat_check_fmt_non_negative_show_denom_0, 0, "{}", "0";
        sat_check_fmt_non_negative_show_denom_1, 0, "{:2}", "0";
        sat_check_fmt_non_negative_show_denom_2, 0, "{:02}", "0";
        sat_check_fmt_non_negative_show_denom_3, 0, "{:9}", "0";
        sat_check_fmt_non_negative_show_denom_4, 0, "{:09}", "0";
        sat_check_fmt_non_negative_show_denom_5, 0, "{:10}", " 0";
        sat_check_fmt_non_negative_show_denom_6, 0, "{:010}", "00";
        sat_check_fmt_non_negative_show_denom_7, 0, "{:.1}", "0.0";
        sat_check_fmt_non_negative_show_denom_8, 0, "{:11.1}", "0.0";
        sat_check_fmt_non_negative_show_denom_9, 0, "{:011.1}", "0.0";
        sat_check_fmt_non_negative_show_denom_10, 0, "{:12.1}", " 0.0";
        sat_check_fmt_non_negative_show_denom_11, 0, "{:012.1}", "00.0";
        sat_check_fmt_non_negative_show_denom_12, 1, "{}", "1";
        sat_check_fmt_non_negative_show_denom_13, 1, "{:10}", " 1";
        sat_check_fmt_non_negative_show_denom_14, 1, "{:010}", "01";
        sat_check_fmt_non_negative_show_denom_15, 1, "{:.1}", "1.0";
        sat_check_fmt_non_negative_show_denom_16, 1, "{:12.1}", " 1.0";
        sat_check_fmt_non_negative_show_denom_17, 1, "{:012.1}", "01.0";
        sat_check_fmt_non_negative_show_denom_18, 10, "{}", "10";
        sat_check_fmt_non_negative_show_denom_19, 10, "{:10}", "10";
        sat_check_fmt_non_negative_show_denom_20, 10, "{:010}", "10";
        sat_check_fmt_non_negative_show_denom_21, 10, "{:11}", " 10";
        sat_check_fmt_non_negative_show_denom_22, 10, "{:011}", "010";
    }

    check_format_non_negative! {
        KAON;
        kaon_check_fmt_non_negative_0, 0, "{}", "0";
        kaon_check_fmt_non_negative_1, 0, "{:2}", " 0";
        kaon_check_fmt_non_negative_2, 0, "{:02}", "00";
        kaon_check_fmt_non_negative_3, 0, "{:.1}", "0.0";
        kaon_check_fmt_non_negative_4, 0, "{:4.1}", " 0.0";
        kaon_check_fmt_non_negative_5, 0, "{:04.1}", "00.0";
        kaon_check_fmt_non_negative_6, 1, "{}", "0.000000000000000001";
        kaon_check_fmt_non_negative_7, 1, "{:2}", "0.000000000000000001";
        kaon_check_fmt_non_negative_8, 1, "{:02}", "0.000000000000000001";
        kaon_check_fmt_non_negative_9, 1, "{:.1}", "0.000000000000000001";
        kaon_check_fmt_non_negative_10, 1, "{:11}", " 0.000000000000000001";
        kaon_check_fmt_non_negative_11, 1, "{:11.1}", " 0.000000000000000001";
        kaon_check_fmt_non_negative_12, 1, "{:011.1}", "00.000000000000000001";
        kaon_check_fmt_non_negative_13, 1, "{:.19}", "0.0000000000000000010";
        kaon_check_fmt_non_negative_14, 1, "{:11.19}", "0.0000000000000000010";
        kaon_check_fmt_non_negative_15, 1, "{:011.19}", "0.0000000000000000010";
        kaon_check_fmt_non_negative_16, 1, "{:12.19}", " 0.0000000000000000010";
        kaon_check_fmt_non_negative_17, 1, "{:012.19}", "00.0000000000000000010";
        kaon_check_fmt_non_negative_18, 1_000_000_000_000_000_000, "{}", "1";
        kaon_check_fmt_non_negative_19, 1_000_000_000_000_000_000, "{:2}", " 1";
        kaon_check_fmt_non_negative_20, 1_000_000_000_000_000_000, "{:02}", "01";
        kaon_check_fmt_non_negative_21, 1_000_000_000_000_000_000, "{:.1}", "1.0";
        kaon_check_fmt_non_negative_22, 1_000_000_000_000_000_000, "{:4.1}", " 1.0";
        kaon_check_fmt_non_negative_23, 1_000_000_000_000_000_000, "{:04.1}", "01.0";
        kaon_check_fmt_non_negative_24, 1_000_000_000_000_000_000, "{}", "1.1";
        kaon_check_fmt_non_negative_25, 1_000_000_010_000_000_000, "{}", "1.00000001";
        kaon_check_fmt_non_negative_26, 1_000_000_010_000_000_000, "{:1}", "1.00000001";
        kaon_check_fmt_non_negative_27, 1_000_000_010_000_000_000, "{:.1}", "1.00000001";
        kaon_check_fmt_non_negative_28, 1_000_000_010_000_000_000, "{:10}", "1.00000001";
        kaon_check_fmt_non_negative_29, 1_000_000_010_000_000_000, "{:11}", " 1.00000001";
        kaon_check_fmt_non_negative_30, 1_000_000_010_000_000_000, "{:011}", "01.00000001";
        kaon_check_fmt_non_negative_31, 1_000_000_010_000_000_000, "{:.8}", "1.00000001";
        kaon_check_fmt_non_negative_32, 1_000_000_010_000_000_000, "{:.9}", "1.000000010";
        kaon_check_fmt_non_negative_33, 1_000_000_010_000_000_000, "{:11.9}", "1.000000010";
        kaon_check_fmt_non_negative_34, 1_000_000_010_000_000_000, "{:12.9}", " 1.000000010";
        kaon_check_fmt_non_negative_35, 1_000_000_010_000_000_000, "{:012.9}", "01.000000010";
        kaon_check_fmt_non_negative_36, 1_000_000_010_000_000_000, "{:+011.8}", "+1.00000001";
        kaon_check_fmt_non_negative_37, 1_000_000_010_000_000_000, "{:+12.8}", " +1.00000001";
        kaon_check_fmt_non_negative_38, 1_000_000_010_000_000_000, "{:+012.8}", "+01.00000001";
        kaon_check_fmt_non_negative_39, 1_000_000_010_000_000_000, "{:+12.9}", "+1.000000010";
        kaon_check_fmt_non_negative_40, 1_000_000_010_000_000_000, "{:+012.9}", "+1.000000010";
        kaon_check_fmt_non_negative_41, 1_000_000_010_000_000_000, "{:+13.9}", " +1.000000010";
        kaon_check_fmt_non_negative_42, 1_000_000_010_000_000_000, "{:+013.9}", "+01.000000010";
        kaon_check_fmt_non_negative_43, 1_000_000_010_000_000_000, "{:<10}", "1.00000001";
        kaon_check_fmt_non_negative_44, 1_000_000_010_000_000_000, "{:<11}", "1.00000001 ";
        kaon_check_fmt_non_negative_45, 1_000_000_010_000_000_000, "{:<011}", "01.00000001";
        kaon_check_fmt_non_negative_46, 1_000_000_010_000_000_000, "{:<11.9}", "1.000000010";
        kaon_check_fmt_non_negative_47, 1_000_000_010_000_000_000, "{:<12.9}", "1.000000010 ";
        kaon_check_fmt_non_negative_48, 1_000_000_010_000_000_000, "{:<12}", "1.00000001  ";
        kaon_check_fmt_non_negative_49, 1_000_000_010_000_000_000, "{:^11}", "1.00000001 ";
        kaon_check_fmt_non_negative_50, 1_000_000_010_000_000_000, "{:^11.9}", "1.000000010";
        kaon_check_fmt_non_negative_51, 1_000_000_010_000_000_000, "{:^12.9}", "1.000000010 ";
        kaon_check_fmt_non_negative_52, 1_000_000_010_000_000_000, "{:^12}", " 1.00000001 ";
        kaon_check_fmt_non_negative_53, 1_000_000_010_000_000_000, "{:^12.9}", "1.000000010 ";
        kaon_check_fmt_non_negative_54, 1_000_000_010_000_000_000, "{:^13.9}", " 1.000000010 ";
    }

    check_format_non_negative_show_denom! {
        KAON, " KAON";
        kaon_check_fmt_non_negative_show_denom_0, 1, "{:14.1}", "0.000000000000000001";
        kaon_check_fmt_non_negative_show_denom_1, 1, "{:14.18}", "0.000000000000000001";
        kaon_check_fmt_non_negative_show_denom_2, 1, "{:15}", " 0.000000000000000001";
        kaon_check_fmt_non_negative_show_denom_3, 1, "{:015}", "00.000000000000000001";
        kaon_check_fmt_non_negative_show_denom_4, 1, "{:.19}", "0.0000000000000000010";
        kaon_check_fmt_non_negative_show_denom_5, 1, "{:15.19}", "0.0000000000000000010";
        kaon_check_fmt_non_negative_show_denom_6, 1, "{:16.19}", " 0.0000000000000000010";
        kaon_check_fmt_non_negative_show_denom_7, 1, "{:016.19}", "00.0000000000000000010";
    }

    check_format_non_negative_show_denom! {
        KAON, " KAON ";
        kaon_check_fmt_non_negative_show_denom_align_0, 1, "{:<15}", "0.000000000000000001";
        kaon_check_fmt_non_negative_show_denom_align_1, 1, "{:^15}", "0.000000000000000001";
        kaon_check_fmt_non_negative_show_denom_align_2, 1, "{:^16}", " 0.000000000000000001";
    }

    check_format_non_negative! {
        PicoSatoshi;
        msat_check_fmt_non_negative_0, 0, "{}", "0";
        msat_check_fmt_non_negative_1, 1, "{}", "100";
        msat_check_fmt_non_negative_2, 1, "{:5}", " 100";
        msat_check_fmt_non_negative_3, 1, "{:05}", "00100";
        msat_check_fmt_non_negative_4, 1, "{:.1}", "100.0";
        msat_check_fmt_non_negative_5, 1, "{:6.1}", "100.0";
        msat_check_fmt_non_negative_6, 1, "{:06.1}", "100.0";
        msat_check_fmt_non_negative_7, 1, "{:7.1}", "  100.0";
        msat_check_fmt_non_negative_8, 1, "{:07.1}", "00100.0";
    }

    #[test]
    fn test_unsigned_signed_conversion() {
        let sa = SignedAmount::from_sat;
        let ua = Amount::from_sat;

        assert_eq!(Amount::MAX.to_signed(), Err(OutOfRangeError::too_big(true)));
        assert_eq!(ua(i128::MAX as u128).to_signed(), Ok(sa(i128::MAX)));
        assert_eq!(ua(i128::MAX as u128 + 1).to_signed(), Err(OutOfRangeError::too_big(true)));

        assert_eq!(sa(i128::MAX).to_unsigned(), Ok(ua(i128::MAX as u128)));

        assert_eq!(sa(i128::MAX).to_unsigned().unwrap().to_signed(), Ok(sa(i128::MAX)));
    }

    #[test]
    #[allow(clippy::inconsistent_digit_grouping)] // Group to show 100,000,000 sats per KAON.
    fn from_str() {
        use ParseDenominationError::*;

        use super::ParseAmountError as E;

        assert_eq!(
            Amount::from_str("x KAON"),
            Err(InvalidCharacterError { invalid_char: 'x', position: 0 }.into())
        );
        assert_eq!(
            Amount::from_str("xKAON"),
            Err(Unknown(UnknownDenominationError("xKAON".into())).into()),
        );
        assert_eq!(
            Amount::from_str("5 KAON KAON"),
            Err(Unknown(UnknownDenominationError("KAON KAON".into())).into()),
        );
        assert_eq!(
            Amount::from_str("5KAON KAON"),
            Err(E::from(InvalidCharacterError { invalid_char: 'B', position: 1 }).into())
        );
        assert_eq!(
            Amount::from_str("5 5 KAON"),
            Err(Unknown(UnknownDenominationError("5 KAON".into())).into()),
        );

        #[track_caller]
        fn ok_case(s: &str, expected: Amount) {
            assert_eq!(Amount::from_str(s).unwrap(), expected);
            assert_eq!(Amount::from_str(&s.replace(' ', "")).unwrap(), expected);
        }

        #[track_caller]
        fn case(s: &str, expected: Result<Amount, impl Into<ParseError>>) {
            let expected = expected.map_err(Into::into);
            assert_eq!(Amount::from_str(s), expected);
            assert_eq!(Amount::from_str(&s.replace(' ', "")), expected);
        }

        #[track_caller]
        fn ok_scase(s: &str, expected: SignedAmount) {
            assert_eq!(SignedAmount::from_str(s).unwrap(), expected);
            assert_eq!(SignedAmount::from_str(&s.replace(' ', "")).unwrap(), expected);
        }

        #[track_caller]
        fn scase(s: &str, expected: Result<SignedAmount, impl Into<ParseError>>) {
            let expected = expected.map_err(Into::into);
            assert_eq!(SignedAmount::from_str(s), expected);
            assert_eq!(SignedAmount::from_str(&s.replace(' ', "")), expected);
        }

        case("5 BCH", Err(Unknown(UnknownDenominationError("BCH".into()))));

        case("-1 KAON", Err(OutOfRangeError::negative()));
        case("-0.0 KAON", Err(OutOfRangeError::negative()));
        case("0.0000000000123456789 KAON", Err(TooPreciseError { position: 20 }));
        scase("-0.1 psat", Err(TooPreciseError { position: 3 }));
        case("0.000000000123456 mKAON", Err(TooPreciseError { position: 16 }));
        scase("-1.0000000000001 bits", Err(TooPreciseError { position: 15 }));
        scase("-2000000000000000000000 KAON", Err(OutOfRangeError::too_small())); // TODO: ensure
        case("184467440737095516160000000000 sat", Err(OutOfRangeError::too_big(false))); // TODO: ensure

        ok_case(".00000000005 bits", Amount::from_sat(50));
        ok_scase("-.00000000005 bits", SignedAmount::from_sat(-50));
        ok_case("0.000000000000253583 KAON", Amount::from_sat(253583));
        ok_scase("-5 satoshi", SignedAmount::from_sat(-50000000000));
        ok_case("0.000000000010000000 KAON", Amount::from_sat(100_000_00));
        ok_scase("-100 bits", SignedAmount::from_sat(-100_000_000_000_000));
        #[cfg(feature = "alloc")]
        ok_scase(&format!("{} aKAON", i128::MIN), SignedAmount::from_sat(i128::MIN));
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[allow(clippy::inconsistent_digit_grouping)] // Group to show 100,000,000 sats per KAON.
    fn to_from_string_in() {
        use super::Denomination as D;
        let ua_str = Amount::from_str_in;
        let ua_sat = Amount::from_sat;
        let sa_str = SignedAmount::from_str_in;
        let sa_sat = SignedAmount::from_sat;

        assert_eq!("0.00000000005", Amount::from_sat(50).to_string_in(D::Bit));
        assert_eq!("-0.00000000005", SignedAmount::from_sat(-50).to_string_in(D::Bit));
        assert_eq!("0.000000000000253583", Amount::from_sat(253583).to_string_in(D::KAON));
        assert_eq!("-5", SignedAmount::from_sat(-50_000_000_000).to_string_in(D::Satoshi));
        assert_eq!("0.00000000001", Amount::from_sat(100_000_00).to_string_in(D::KAON));
        assert_eq!("-100", SignedAmount::from_sat(-100_000_000_000_000).to_string_in(D::Bit));
        assert_eq!("2535830", Amount::from_sat(2535830_000_000_000).to_string_in(D::NanoKAON));
        assert_eq!("0.10000000", format!("{:.8}", Amount::from_sat(100_000_000_000_000_000).display_in(D::KAON)));
        assert_eq!("-100000",SignedAmount::from_sat(-100_000_000_000_000).to_string_in(D::NanoKAON));
        assert_eq!("2535830000", Amount::from_sat(2535830_000_000_000).to_string_in(D::PicoKAON));
        assert_eq!("-100000000", SignedAmount::from_sat(-1000000000000000).to_string_in(D::PicoKAON));

        assert_eq!("0.000000000050", format!("{:.12}", Amount::from_sat(50).display_in(D::Bit)));
        assert_eq!("-0.000000000050", format!("{:.12}", SignedAmount::from_sat(-50).display_in(D::Bit)));
        assert_eq!(
            "0.10000000",
            format!("{:.8}", Amount::from_sat(100_000_000_000_000_000).display_in(D::KAON))
        );
        assert_eq!("-100.00", format!("{:.2}", SignedAmount::from_sat(-100_000_000_000_000).display_in(D::Bit)));

        assert_eq!(ua_str(&ua_sat(0).to_string_in(D::Satoshi), D::Satoshi), Ok(ua_sat(0)));
        assert_eq!(ua_str(&ua_sat(500).to_string_in(D::KAON), D::KAON), Ok(ua_sat(500)));
        assert_eq!(
            ua_str(&ua_sat(21_000_000_000_000_000_000).to_string_in(D::Bit), D::Bit),
            Ok(ua_sat(21_000_000_000_000_000_000))
        );
        assert_eq!(
            ua_str(&ua_sat(1).to_string_in(D::MicroKAON), D::MicroKAON), 
            Ok(ua_sat(1))
        );
        assert_eq!(
            ua_str(&ua_sat(1_000_000_000_000).to_string_in(D::MilliKAON), D::MilliKAON),
            Ok(ua_sat(1_000_000_000_000))
        );
        assert!(ua_str(&ua_sat(u128::MAX).to_string_in(D::MilliKAON), D::MilliKAON).is_ok());

        assert_eq!(
            sa_str(&sa_sat(-1).to_string_in(D::MicroKAON), D::MicroKAON), 
            Ok(sa_sat(-1))
        );

        assert_eq!(
            sa_str(&sa_sat(i128::MAX).to_string_in(D::Satoshi), D::MicroKAON),
            Err(OutOfRangeError::too_big(true).into())
        );
        // Test an overflow bug in `abs()`
        assert_eq!(
            sa_str(&sa_sat(i128::MIN).to_string_in(D::Satoshi), D::MicroKAON),
            Err(OutOfRangeError::too_small().into())
        );

        assert_eq!(
            sa_str(&sa_sat(-1).to_string_in(D::NanoKAON), D::NanoKAON), 
            Ok(sa_sat(-1))
        );
        assert_eq!(
            sa_str(&sa_sat(i128::MAX).to_string_in(D::Satoshi), D::NanoKAON),
            Err(TooPreciseError { position: 18 }.into())
        );
        assert_eq!(
            sa_str(&sa_sat(i128::MIN).to_string_in(D::Satoshi), D::NanoKAON),
            Err(TooPreciseError { position: 19 }.into())
        );

        assert_eq!(
            sa_str(&sa_sat(-1).to_string_in(D::PicoKAON), D::PicoKAON), 
            Ok(sa_sat(-1))
        );
        assert_eq!(
            sa_str(&sa_sat(i128::MAX).to_string_in(D::Satoshi), D::PicoKAON),
            Err(TooPreciseError { position: 18 }.into())
        );
        assert_eq!(
            sa_str(&sa_sat(i128::MIN).to_string_in(D::Satoshi), D::PicoKAON),
            Err(TooPreciseError { position: 19 }.into())
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn to_string_with_denomination_from_str_roundtrip() {
        use ParseDenominationError::*;

        use super::Denomination as D;

        let amt = Amount::from_sat(42);
        let denom = Amount::to_string_with_denomination;
        assert_eq!(Amount::from_str(&denom(amt, D::KAON)), Ok(amt));
        assert_eq!(Amount::from_str(&denom(amt, D::MilliKAON)), Ok(amt));
        assert_eq!(Amount::from_str(&denom(amt, D::MicroKAON)), Ok(amt));
        assert_eq!(Amount::from_str(&denom(amt, D::Bit)), Ok(amt));
        assert_eq!(Amount::from_str(&denom(amt, D::Satoshi)), Ok(amt));
        assert_eq!(Amount::from_str(&denom(amt, D::NanoKAON)), Ok(amt));
        assert_eq!(Amount::from_str(&denom(amt, D::MilliSatoshi)), Ok(amt));
        assert_eq!(Amount::from_str(&denom(amt, D::PicoKAON)), Ok(amt));

        assert_eq!(
            Amount::from_str("42 satoshi KAON"),
            Err(Unknown(UnknownDenominationError("satoshi KAON".into())).into()),
        );
        assert_eq!(
            SignedAmount::from_str("-42 satoshi KAON"),
            Err(Unknown(UnknownDenominationError("satoshi KAON".into())).into()),
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_as_sat() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct T {
            #[serde(with = "crate::amount::serde::as_sat")]
            pub amt: Amount,
            #[serde(with = "crate::amount::serde::as_sat")]
            pub samt: SignedAmount,
        }

        serde_test::assert_tokens(
            &T { amt: Amount::from_sat(123456789), samt: SignedAmount::from_sat(-123456789) },
            &[
                serde_test::Token::Struct { name: "T", len: 2 },
                serde_test::Token::Str("amt"),
                serde_test::Token::U64(123456789), // TODO: add U128
                serde_test::Token::Str("samt"),
                serde_test::Token::I64(-123456789), // TODO: add I128
                serde_test::Token::StructEnd,
            ],
        );
    }

    #[cfg(feature = "serde")]
    #[cfg(feature = "alloc")]
    #[test]
    #[allow(clippy::inconsistent_digit_grouping)] // Group to show 100,000,000...0 sats per KAON.
    fn serde_as_kaon() {
        use serde_json;

        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct T {
            #[serde(with = "crate::amount::serde::as_kaon")]
            pub amt: Amount,
            #[serde(with = "crate::amount::serde::as_kaon")]
            pub samt: SignedAmount,
        }

        let orig = T {
            amt: Amount::from_sat(21_000_000__000_000_010_000_000_000),
            samt: SignedAmount::from_sat(-21_000_000__000_000_010_000_000_000),
        };

        let json = "{\"amt\": 21000000.00000001, \
                    \"samt\": -21000000.00000001}";
        let t: T = serde_json::from_str(json).unwrap();
        assert_eq!(t, orig);

        let value: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(t, serde_json::from_value(value).unwrap());

        // errors
        let t: Result<T, serde_json::Error> =
            serde_json::from_str("{\"amt\": 1000000.00000000000000001, \"samt\": 1}");
        assert!(t
            .unwrap_err()
            .to_string()
            .contains(&ParseAmountError::TooPrecise(TooPreciseError { position: 24 }).to_string()));
        let t: Result<T, serde_json::Error> = serde_json::from_str("{\"amt\": -1, \"samt\": 1}");
        assert!(t.unwrap_err().to_string().contains(&OutOfRangeError::negative().to_string()));
    }

    #[cfg(feature = "serde")]
    #[cfg(feature = "alloc")]
    #[test]
    #[allow(clippy::inconsistent_digit_grouping)] // Group to show 100,000,000 sats per KAON.
    fn serde_as_kaon_opt() {
        use serde_json;

        #[derive(Serialize, Deserialize, PartialEq, Debug, Eq)]
        struct T {
            #[serde(default, with = "crate::amount::serde::as_kaon::opt")]
            pub amt: Option<Amount>,
            #[serde(default, with = "crate::amount::serde::as_kaon::opt")]
            pub samt: Option<SignedAmount>,
        }

        let with = T {
            amt: Some(Amount::from_sat(2_500_000_000_000_000_000)),
            samt: Some(SignedAmount::from_sat(-2_500_000_000_000_000_000)),
        };
        let without = T { amt: None, samt: None };

        // Test Roundtripping
        for s in [&with, &without].iter() {
            let v = serde_json::to_string(s).unwrap();
            let w: T = serde_json::from_str(&v).unwrap();
            assert_eq!(w, **s);
        }

        let t: T = serde_json::from_str("{\"amt\": 2.5, \"samt\": -2.5}").unwrap();
        assert_eq!(t, with);

        let t: T = serde_json::from_str("{}").unwrap();
        assert_eq!(t, without);

        let value_with: serde_json::Value =
            serde_json::from_str("{\"amt\": 2.5, \"samt\": -2.5}").unwrap();
        assert_eq!(with, serde_json::from_value(value_with).unwrap());

        let value_without: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert_eq!(without, serde_json::from_value(value_without).unwrap());
    }

    #[cfg(feature = "serde")]
    #[cfg(feature = "alloc")]
    #[test]
    #[allow(clippy::inconsistent_digit_grouping)] // Group to show 100,000,000 sats per KAON.
    fn serde_as_sat_opt() {
        use serde_json;

        #[derive(Serialize, Deserialize, PartialEq, Debug, Eq)]
        struct T {
            #[serde(default, with = "crate::amount::serde::as_sat::opt")]
            pub amt: Option<Amount>,
            #[serde(default, with = "crate::amount::serde::as_sat::opt")]
            pub samt: Option<SignedAmount>,
        }

        let with = T {
            amt: Some(Amount::from_sat(2_500_000_000_000_000_000)),
            samt: Some(SignedAmount::from_sat(-2_500_000_000_000_000_000)),
        };
        let without = T { amt: None, samt: None };

        // Test Roundtripping
        for s in [&with, &without].iter() {
            let v = serde_json::to_string(s).unwrap();
            let w: T = serde_json::from_str(&v).unwrap();
            assert_eq!(w, **s);
        }

        let t: T = serde_json::from_str("{\"amt\": 2500000000000000000, \"samt\": -2500000000000000000}").unwrap();
        assert_eq!(t, with);

        let t: T = serde_json::from_str("{}").unwrap();
        assert_eq!(t, without);

        let value_with: serde_json::Value =
            serde_json::from_str("{\"amt\": 2500000000000000000, \"samt\": -2500000000000000000}").unwrap();
        assert_eq!(with, serde_json::from_value(value_with).unwrap());

        let value_without: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert_eq!(without, serde_json::from_value(value_without).unwrap());
    }

    #[test]
    fn sum_amounts() {
        assert_eq!(Amount::from_sat(0), [].into_iter().sum::<Amount>());
        assert_eq!(SignedAmount::from_sat(0), [].into_iter().sum::<SignedAmount>());

        let amounts = [Amount::from_sat(42), Amount::from_sat(1337), Amount::from_sat(21)];
        let sum = amounts.into_iter().sum::<Amount>();
        assert_eq!(Amount::from_sat(1400), sum);

        let amounts =
            [SignedAmount::from_sat(-42), SignedAmount::from_sat(1337), SignedAmount::from_sat(21)];
        let sum = amounts.into_iter().sum::<SignedAmount>();
        assert_eq!(SignedAmount::from_sat(1316), sum);
    }

    #[test]
    fn checked_sum_amounts() {
        assert_eq!(Some(Amount::from_sat(0)), [].into_iter().checked_sum());
        assert_eq!(Some(SignedAmount::from_sat(0)), [].into_iter().checked_sum());

        let amounts = [Amount::from_sat(42), Amount::from_sat(1337), Amount::from_sat(21)];
        let sum = amounts.into_iter().checked_sum();
        assert_eq!(Some(Amount::from_sat(1400)), sum);

        let amounts = [Amount::from_sat(u128::MAX), Amount::from_sat(1337), Amount::from_sat(21)];
        let sum = amounts.into_iter().checked_sum();
        assert_eq!(None, sum);

        let amounts = [
            SignedAmount::from_sat(i128::MIN),
            SignedAmount::from_sat(-1),
            SignedAmount::from_sat(21),
        ];
        let sum = amounts.into_iter().checked_sum();
        assert_eq!(None, sum);

        let amounts = [
            SignedAmount::from_sat(i128::MAX),
            SignedAmount::from_sat(1),
            SignedAmount::from_sat(21),
        ];
        let sum = amounts.into_iter().checked_sum();
        assert_eq!(None, sum);

        let amounts =
            [SignedAmount::from_sat(42), SignedAmount::from_sat(3301), SignedAmount::from_sat(21)];
        let sum = amounts.into_iter().checked_sum();
        assert_eq!(Some(SignedAmount::from_sat(3364)), sum);
    }

    #[test]
    fn denomination_string_acceptable_forms() {
        // Non-exhaustive list of valid forms.
        let valid = [
            "KAON", "kaon", "mKAON", "mkaon", "uKAON", "ukaon", "SATOSHI", "satoshi", "SATOSHIS",
            "satoshis", "SAT", "sat", "SATS", "sats", "bit", "bits", "nKAON", "pKAON",
        ];
        for denom in valid.iter() {
            assert!(Denomination::from_str(denom).is_ok());
        }
    }

    #[test]
    fn disallow_confusing_forms() {
        let confusing = ["Msat", "Msats", "MSAT", "MSATS", "MSat", "MSats", "MBTC", "Mbtc", "PBTC", "MKAON", "Mkaon", "PKAON"];
        for denom in confusing.iter() {
            match Denomination::from_str(denom) {
                Ok(_) => panic!("from_str should error for {}", denom),
                Err(ParseDenominationError::PossiblyConfusing(_)) => {}
                Err(e) => panic!("unexpected error: {}", e),
            }
        }
    }

    #[test]
    fn disallow_unknown_denomination() {
        // Non-exhaustive list of unknown forms.
        let unknown = ["BTC", "NBTC", "UBTC", "NKAON", "UKAON", "ABC", "abc", "cBtC", "cKAON", "Sat", "Sats"];
        for denom in unknown.iter() {
            match Denomination::from_str(denom) {
                Ok(_) => panic!("from_str should error for {}", denom),
                Err(ParseDenominationError::Unknown(_)) => (),
                Err(e) => panic!("unexpected error: {}", e),
            }
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn trailing_zeros_for_amount() {
        assert_eq!(format!("{}", Amount::ONE_AKAON), "0.000000000000000001 KAON");
        assert_eq!(format!("{}", Amount::ONE_KAON), "1 KAON");
        assert_eq!(format!("{}", Amount::from_sat(1)), "0.000000000000000001 KAON");
        assert_eq!(format!("{}", Amount::from_sat(10)), "0.000000000000000010 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(10)), "0.00000000000000001 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(100)), "0.0000000000000001 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(1000)), "0.000000000000001 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(10_000)), "0.0000000000001 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(100_000)), "0.000000000001 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(1_000_000)), "0.00000000001 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(10_000_000)), "0.0000000001 KAON");
        assert_eq!(format!("{:.2}", Amount::from_sat(1_000_000_000_000_000_000)), "1.00 KAON");
        assert_eq!(format!("{}", Amount::from_sat(1_000_000_000_000_000_000)), "1 KAON");
        assert_eq!(format!("{}", Amount::from_sat(400_000_000_000_000_000_000)), "400 KAON");
        assert_eq!(format!("{:.10}", Amount::from_sat(1_000_000_000_000_000_000)), "1.0000000000 KAON");
        assert_eq!(format!("{}", Amount::from_sat(4_000_000_000_000_000_010_000_000)), "4000000.00000010 KAON");
        assert_eq!(format!("{}", Amount::from_sat(4_000_000_000_000_000_000_000_000)),"4000000 KAON");
    }
}
