// SPDX-License-Identifier: CC0-1.0

//! Bitcoin/Kaon consensus parameters.
//!
//! This module provides a predefined set of parameters for different Bitcoin/Kaon
//! chains (such as mainnet, testnet).
//!
//! # Custom Signets Example
//!
//! In various places in this crate we take `AsRef<Params>` as a parameter, in order to create a
//! custom type that can be used is such places you might want to do the following:
//!
//! ```
//! use kaon::consensus::Params;
//! use kaon::{p2p, Script, ScriptBuf, Network, Target};
//!
//! const POW_TARGET_SPACING: u64 = 120; // Two minutes.
//! const MAGIC: [u8; 4] = [1, 2, 3, 4];
//!
//! pub struct CustomParams {
//!     params: Params,
//!     magic: [u8; 4],
//!     challenge_script: ScriptBuf,
//! }
//!
//! impl CustomParams {
//!     /// Creates a new custom params.
//!     pub fn new() -> Self {
//!         let mut params = Params::new(Network::Signet);
//!         params.pow_target_spacing = POW_TARGET_SPACING;
//!
//!         // This would be something real (see BIP-325).
//!         let challenge_script = ScriptBuf::new();
//!
//!         Self {
//!             params,
//!             magic: MAGIC,
//!             challenge_script,
//!         }
//!     }
//!
//!     /// Returns the custom magic bytes.
//!     pub fn magic(&self) -> p2p::Magic { p2p::Magic::from_bytes(self.magic) }
//!
//!     /// Returns the custom signet challenge script.
//!     pub fn challenge_script(&self) -> &Script { &self.challenge_script }
//! }
//!
//! impl AsRef<Params> for CustomParams {
//!     fn as_ref(&self) -> &Params { &self.params }
//! }
//!
//! impl Default for CustomParams {
//!     fn default() -> Self { Self::new() }
//! }
//!
//! # { // Just check the code above is usable.
//! #    let target = Target::MAX_ATTAINABLE_SIGNET;
//! #
//! #    let signet = Params::SIGNET;
//! #    let _ = target.difficulty(signet);
//! #
//! #    let custom = CustomParams::new();
//! #    let _ = target.difficulty(custom);
//! # }
//! ```

use units::{BlockHeight, BlockInterval};

use crate::network::Network;
#[cfg(doc)]
use crate::pow::CompactTarget;
use crate::pow::Target;

/// Parameters that influence chain consensus.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Params {
    // TODO: validate parameters
    /// Network for which parameters are valid.
    pub network: Network,
    /// Time when BIP16 becomes active.
    pub bip16_time: u32,
    /// Block height at which BIP34 becomes active.
    pub bip34_height: BlockHeight,
    /// Block height at which BIP65 becomes active.
    pub bip65_height: BlockHeight,
    /// Block height at which BIP66 becomes active.
    pub bip66_height: BlockHeight,
    /// Minimum blocks including miner confirmation of the total of 2016 blocks in a retargeting period,
    /// (nPowTargetTimespan / nPowTargetSpacing) which is also used for BIP9 deployments.
    /// Examples: 1916 for 95%, 1512 for testchains.
    pub rule_change_activation_threshold: BlockInterval,
    /// Number of blocks with the same set of rules.
    pub miner_confirmation_window: BlockInterval,
    /// Proof of work limit value. It contains the lowest possible difficulty.
    #[deprecated(since = "0.32.0", note = "field renamed to max_attainable_target")]
    pub pow_limit: Target, // TODO: PoS
    /// The maximum **attainable** target value for these params.
    ///
    /// Not all target values are attainable because consensus code uses the compact format to
    /// represent targets (see [`CompactTarget`]).
    ///
    /// Note that this value differs from Bitcoin Core's powLimit field in that this value is
    /// attainable, but Bitcoin Core's is not. Specifically, because targets in Bitcoin are always
    /// rounded to the nearest float expressible in "compact form", not all targets are attainable.
    /// Still, this should not affect consensus as the only place where the non-compact form of
    /// this is used in Bitcoin Core's consensus algorithm is in comparison and there are no
    /// compact-expressible values between Bitcoin Core's and the limit expressed here.
    pub max_attainable_target: Target,
    /// Expected amount of time to mine one block.
    pub pow_target_spacing: u64,
    /// Difficulty recalculation interval.
    pub pow_target_timespan: u64,
    /// Determines whether minimal difficulty may be used for blocks or not.
    pub allow_min_difficulty_blocks: bool,
    /// Determines whether retargeting is disabled for this network or not.
    pub no_pow_retargeting: bool,
}

/// The mainnet parameters.
///
/// Use this for a static reference e.g., `&params::MAINNET`.
///
/// For more on static vs const see The Rust Reference [using-statics-or-consts] section.
///
/// [using-statics-or-consts]: <https://doc.rust-lang.org/reference/items/static-items.html#using-statics-or-consts>
pub static MAINNET: Params = Params::MAINNET;
/// The testnet parameters.
pub static TESTNET: Params = Params::TESTNET;
/// The signet parameters.
pub static SIGNET: Params = Params::SIGNET;
/// The regtest parameters.
pub static REGTEST: Params = Params::REGTEST;

#[allow(deprecated)] // For `pow_limit`.
impl Params {
    /// The mainnet parameters (alias for `Params::MAINNET`).
    pub const KAON: Params = Params::MAINNET;

    // TODO: add chain-specific details, dPOS parameters included
    /// The mainnet parameters.
    pub const MAINNET: Params = Params {
        network: Network::Mainnet,
        bip16_time: 1333238400, // Apr 1 2012
        bip34_height: 1,
        bip65_height: 1,
        bip66_height: 1,
        rule_change_activation_threshold: 10260, // 95%
        miner_confirmation_window: 2016,
        pow_limit: Target::MAX_ATTAINABLE_MAINNET,
        pow_target_spacing: 15,            // 15 seconds.
        pow_target_timespan: 2 * 15 * 60,  // 30 min
        allow_min_difficulty_blocks: true, // POW
        no_pow_retargeting: false,
    };

    /// The testnet parameters.
    pub const TESTNET: Params = Params {
        network: Network::Testnet,
        bip16_time: 1333238400, // Apr 1 2012
        bip34_height: 1,
        bip65_height: 1,
        bip66_height: 1,
        rule_change_activation_threshold: 8100, // 75%
        miner_confirmation_window: 2016,
        pow_limit: Target::MAX_ATTAINABLE_TESTNET,
        pow_target_spacing: 12,            // 12 seconds.
        pow_target_timespan: 2 * 15 * 60,  // 30 min
        allow_min_difficulty_blocks: true, // POW
        no_pow_retargeting: false,
    };

    /// The signet parameters.
    pub const SIGNET: Params = Params {
        network: Network::Signet,
        bip16_time: 1333238400, // Apr 1 2012
        bip34_height: 1,
        bip65_height: 1,
        bip66_height: 1,
        rule_change_activation_threshold: 10260, // 95%
        miner_confirmation_window: 2016,
        pow_limit: Target::MAX_ATTAINABLE_SIGNET,
        pow_target_spacing: 12,            // 12 seconds.
        pow_target_timespan: 2 * 15 * 60,  // 30 min
        allow_min_difficulty_blocks: true, // POW
        no_pow_retargeting: false,
    };

    /// The regtest parameters.
    pub const REGTEST: Params = Params {
        network: Network::Regtest,
        bip16_time: 1333238400, // Apr 1 2012
        bip34_height: 1,
        bip65_height: 1,
        bip66_height: 1,
        rule_change_activation_threshold: 108, // 75%
        miner_confirmation_window: 144,
        pow_limit: Target::MAX_ATTAINABLE_REGTEST,
        pow_target_spacing: 12,            // 12 seconds.
        pow_target_timespan: 2 * 15 * 60,  // 30 min
        allow_min_difficulty_blocks: true, // POW
        no_pow_retargeting: false,
    };

    /// Creates parameters set for the given network.
    pub const fn new(network: Network) -> Self {
        match network {
            Network::Mainnet => Params::MAINNET,
            Network::Testnet => Params::TESTNET,
            Network::Signet => Params::SIGNET,
            Network::Regtest => Params::REGTEST,
        }
    }

    /// Calculates the number of blocks between difficulty adjustments.
    pub fn difficulty_adjustment_interval(&self) -> u64 {
        self.pow_target_timespan / self.pow_target_spacing
    }
}

impl From<Network> for Params {
    fn from(value: Network) -> Self { Self::new(value) }
}

impl From<&Network> for Params {
    fn from(value: &Network) -> Self { Self::new(*value) }
}

impl From<Network> for &'static Params {
    fn from(value: Network) -> Self { value.params() }
}

impl From<&Network> for &'static Params {
    fn from(value: &Network) -> Self { value.params() }
}

impl AsRef<Params> for Params {
    fn as_ref(&self) -> &Params { self }
}

impl AsRef<Params> for Network {
    fn as_ref(&self) -> &Params {
        match *self {
            Network::Mainnet => &MAINNET,
            Network::Testnet => &TESTNET,
            Network::Signet => &SIGNET,
            Network::Regtest => &REGTEST,
        }
    }
}
