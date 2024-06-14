// SPDX-License-Identifier: CC0-1.0

//! Blockdata constants.
//!
//! This module provides various constants relating to the blockchain and
//! consensus code. In particular, it defines the genesis block and its
//! single transaction.

use hashes::{sha256d, Hash};
use internals::impl_array_newtype;

use crate::blockdata::block::{self, Block, BlockStateRoot};
use crate::blockdata::locktime::absolute;
use crate::blockdata::opcodes::all::*;
use crate::blockdata::script;
use crate::blockdata::transaction::{self, OutPoint, Sequence, Transaction, TxIn, TxOut};
use crate::blockdata::witness::Witness;
use crate::consensus::Params;
use crate::crypto::taproot::Signature;
use crate::internal_macros::impl_array_newtype_stringify;
use crate::network::Network;
use crate::pow::CompactTarget;
use crate::transaction::ValidatorRegister;
use crate::{Amount, BlockHash, PublicKey};

/// How many seconds between blocks we expect on average.
pub const TARGET_BLOCK_SPACING: u32 = 15;
/// How many blocks between diffchanges.
pub const DIFFCHANGE_INTERVAL: u32 = 2016; // TODO: calculate more presice values
/// How much time on average should occur between diffchanges.
pub const DIFFCHANGE_TIMESPAN: u32 = 14 * 24 * 3600; // TODO: calculate more presice values

/// The factor that non-witness serialization data is multiplied by during weight calculation.
pub const WITNESS_SCALE_FACTOR: usize = units::weight::WITNESS_SCALE_FACTOR;
/// The maximum allowed number of signature check operations in a block.
pub const MAX_BLOCK_SIGOPS_COST: i64 = 80_000;
/// Mainnet (Kaon) pubkey address prefix.
pub const PUBKEY_ADDRESS_PREFIX_MAIN: u8 = 85; // 0x55
/// Mainnet (Kaon) script address prefix.
pub const SCRIPT_ADDRESS_PREFIX_MAIN: u8 = 20; // 0x14
/// Test (tesnet, signet, regtest) pubkey address prefix.
pub const PUBKEY_ADDRESS_PREFIX_TEST: u8 = 84; // 0x54
/// Test (tesnet, signet, regtest) script address prefix.
pub const SCRIPT_ADDRESS_PREFIX_TEST: u8 = 40; // 0x28
                                               // PUBKEY_ADDRESS_PREFIX_REGTEST: u8 = 82; // 0x52
                                               // SCRIPT_ADDRESS_PREFIX_REGTEST: u8 = 60; // 0x3C
/// The maximum allowed script size.
pub const MAX_SCRIPT_ELEMENT_SIZE: usize = 129_000;
/// How may blocks between halvings.
pub const SUBSIDY_HALVING_INTERVAL: u32 = 985_500;
/// Maximum allowed value for an integer in Script.
pub const MAX_SCRIPTNUM_VALUE: u128 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF; // 2^127-1
/// Number of blocks needed for an output from a coinbase transaction to be spendable.
pub const COINBASE_MATURITY: u32 = 9; // TODO: will be increased later

/// Constructs and returns the coinbase (and only) transaction of the Kaon genesis block.
// This is the 65 byte (uncompressed) pubkey used as the one-and-only output of the genesis transaction.
//
// ref: https://blockstream.info/tx/4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b?expand
// Note output script includes a leading 0x41 and trailing 0xac (added below using the `script::Builder`).
// NOTE: Update it later with ours key
#[rustfmt::skip]
const GENESIS_OUTPUT_PK: [u8; 65] = [
    0x04,
    0x67, 0x8a, 0xfd, 0xb0, 0xfe, 0x55, 0x48, 0x27,
    0x19, 0x67, 0xf1, 0xa6, 0x71, 0x30, 0xb7, 0x10,
    0x5c, 0xd6, 0xa8, 0x28, 0xe0, 0x39, 0x09, 0xa6,
    0x79, 0x62, 0xe0, 0xea, 0x1f, 0x61, 0xde, 0xb6,
    0x49, 0xf6, 0xbc, 0x3f, 0x4c, 0xef, 0x38, 0xc4,
    0xf3, 0x55, 0x04, 0xe5, 0x1e, 0xc1, 0x12, 0xde,
    0x5c, 0x38, 0x4d, 0xf7, 0xba, 0x0b, 0x8d, 0x57,
    0x8a, 0x4c, 0x70, 0x2b, 0x6b, 0xf1, 0x1d, 0x5f
];

/// Constructs and returns the coinbase (and only) transaction of the Kaon genesis block.
// TODO: rename to kaon_genesis_tx
fn bitcoin_genesis_tx() -> Transaction {
    // Base
    let mut ret = Transaction {
        version: transaction::Version::ONE,
        lock_time: absolute::LockTime::ZERO,
        input: vec![],
        output: vec![],
        validator_register: vec![],
        validator_vote: vec![],
        gas_price: Amount::ZERO,
    };

    // Inputs
    let in_script = script::Builder::new()
        .push_int(486604799)
        .push_int_non_minimal(4)
        .push_slice(b"ARPAnet invention or The Internet began with a crash on October 29, 1969")
        .into_script();
    ret.input.push(TxIn {
        previous_output: OutPoint::null(),
        script_sig: in_script,
        sequence: Sequence::MAX,
        witness: Witness::default(),
    });

    ret.validator_register.push(ValidatorRegister {
        public_key: PublicKey::from_str(
            "02f10964b5084147d013e63108c9df67b8cbe1c6402b1948c849dd51a8dcca9e9f",
        )
        .unwrap(),
        // TODO: for the testnet it is "028d13c338d470038e7a9183cf64c11681ba916a99b0261be14449c4c004f157ce"
        // it is important to sapport all these values versions
        vin: TxIn {
            previous_output: OutPoint { txid: Hash::all_zeros(), vout: 0 },
            script_sig: script::ScriptBuf::new(),
            sequence: Sequence::MAX,
            witness: Witness::default(),
        },
        time: 0,
        signature: Signature::default(),
    });

    // Outputs
    let out_script =
        script::Builder::new().push_slice(GENESIS_OUTPUT_PK).push_opcode(OP_CHECKSIG).into_script();
    ret.output.push(TxOut { value: Amount::from_akaon(50 * 1_000_000_000_000_000_000), script_pubkey: out_script });

    // end
    ret
}

/// Constructs and returns the genesis block.
pub fn genesis_block(params: impl AsRef<Params>) -> Block {
    let txdata = vec![bitcoin_genesis_tx()];
    let hash: sha256d::Hash = txdata[0].compute_txid().into();
    let merkle_root = hash.into();
    match params.as_ref().network {
        Network::Mainnet => Block {
            header: block::Header {
                version: block::Version::ONE,
                prev_blockhash: BlockHash::all_zeros(),
                merkle_root,
                time: 1638961350,
                bits: CompactTarget::from_consensus(0x1E0FFFF0),
                nonce: 607505,
                // TODO: ensure endian of the value
                hash_state_root: BlockStateRoot::from_str(
                    "e965ffd002cd6ad0e2dc402b8044de833e06b23127ea8c3d80aec91410771495",
                )
                .unwrap(),
                // TODO: ensure that RLP from empty value is producing exactly all zeroes
                hash_utxo_root: Hash::all_zeros(),
                gas_used: Amount::ZERO,
            },
            txdata,
        },
        Network::Testnet => Block {
            header: block::Header {
                version: block::Version::ONE,
                prev_blockhash: BlockHash::all_zeros(),
                merkle_root,
                time: 1504695028,
                bits: CompactTarget::from_consensus(0x1f00ffff),
                nonce: 8026361,
                // TODO: ensure endian of the value
                hash_state_root: BlockStateRoot::from_str(
                    "e965ffd002cd6ad0e2dc402b8044de833e06b23127ea8c3d80aec91410771495",
                )
                .unwrap(),
                // TODO: ensure that RLP from empty value is producing exactly all zeroes
                hash_utxo_root: Hash::all_zeros(),
                gas_used: Amount::ZERO,
            },
            txdata,
        },
        Network::Signet => Block {
            header: block::Header {
                version: block::Version::ONE,
                prev_blockhash: BlockHash::all_zeros(),
                merkle_root,
                time: 1598918400,
                bits: CompactTarget::from_consensus(0x1e0377ae),
                nonce: 52613770,
                // TODO: ensure endian of the value
                hash_state_root: BlockStateRoot::from_str(
                    "e965ffd002cd6ad0e2dc402b8044de833e06b23127ea8c3d80aec91410771495",
                )
                .unwrap(),
                // TODO: ensure that RLP from empty value is producing exactly all zeroes
                hash_utxo_root: Hash::all_zeros(),
                gas_used: Amount::ZERO,
            },
            txdata,
        },
        Network::Regtest => Block {
            header: block::Header {
                version: block::Version::ONE,
                prev_blockhash: BlockHash::all_zeros(),
                merkle_root,
                time: 1504695029,
                bits: CompactTarget::from_consensus(0x1f00ffff),
                nonce: 8026361,
                // TODO: ensure endian of the value
                hash_state_root: BlockStateRoot::from_str(
                    "e965ffd002cd6ad0e2dc402b8044de833e06b23127ea8c3d80aec91410771495",
                )
                .unwrap(),
                // TODO: ensure that RLP from empty value is producing exactly all zeroes
                hash_utxo_root: Hash::all_zeros(),
                gas_used: Amount::ZERO,
            },
            txdata,
        },
    }
}

/// The uniquely identifying hash of the target blockchain.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainHash([u8; 32]);
impl_array_newtype!(ChainHash, u8, 32);
impl_array_newtype_stringify!(ChainHash, 32);

impl ChainHash {
    // Mainnet is not launched yet
    // TODO: update the value when it'll be done
    /// `ChainHash` for mainnet Kaon.
    pub const MAINNET: Self = Self([
        130, 208, 182, 228, 190, 39, 139, 58, 209, 174, 231, 210, 220, 19, 37, 62, 228, 2, 43, 6,
        229, 147, 72, 49, 180, 160, 167, 15, 23, 105, 252, 78,
    ]);
    /// `ChainHash` for testnet Kaon.
    pub const TESTNET: Self = Self([
        101, 103, 70, 236, 56, 34, 7, 247, 232, 204, 119, 104, 28, 218, 103, 199, 66, 161, 161,
        111, 160, 106, 147, 1, 186, 31, 86, 174, 219, 216, 82, 22,
    ]);
    /// `ChainHash` for signet Kaon.
    pub const SIGNET: Self = Self([
        209, 150, 203, 81, 113, 143, 151, 226, 130, 246, 190, 44, 69, 46, 221, 254, 14, 152, 78,
        255, 241, 162, 178, 248, 90, 86, 96, 120, 81, 243, 91, 211,
    ]);
    /// `ChainHash` for regtest Kaon.
    pub const REGTEST: Self = Self([
        209, 150, 203, 81, 113, 143, 151, 226, 130, 246, 190, 44, 69, 46, 221, 254, 14, 152, 78,
        255, 241, 162, 178, 248, 90, 86, 96, 120, 81, 243, 91, 211,
    ]);

    /// Returns the hash of the `network` genesis block for use as a chain hash.
    ///
    /// See [BOLT 0](https://github.com/lightning/bolts/blob/ffeece3dab1c52efdb9b53ae476539320fa44938/00-introduction.md#chain_hash)
    /// for specification.
    pub fn using_genesis_block(params: impl AsRef<Params>) -> Self {
        let network = params.as_ref().network;
        let hashes = [Self::MAINNET, Self::TESTNET, Self::SIGNET, Self::REGTEST];
        hashes[network as usize]
    }

    /// Returns the hash of the `network` genesis block for use as a chain hash.
    ///
    /// See [BOLT 0](https://github.com/lightning/bolts/blob/ffeece3dab1c52efdb9b53ae476539320fa44938/00-introduction.md#chain_hash)
    /// for specification.
    pub const fn using_genesis_block_const(network: Network) -> Self {
        let hashes = [Self::MAINNET, Self::TESTNET, Self::SIGNET, Self::REGTEST];
        hashes[network as usize]
    }

    /// Converts genesis block hash into `ChainHash`.
    pub fn from_genesis_block_hash(block_hash: crate::BlockHash) -> Self {
        ChainHash(block_hash.to_byte_array())
    }
}

#[cfg(test)]
mod test {
    use core::str::FromStr;

    use hex::test_hex_unwrap as hex;

    use super::*;
    use crate::consensus::encode::serialize;
    use crate::consensus::params;
    use crate::Txid;

    #[test]
    fn bitcoin_genesis_first_transaction() {
        let gen = bitcoin_genesis_tx();

        assert_eq!(gen.version, transaction::Version::ONE);
        assert_eq!(gen.input.len(), 1);
        assert_eq!(gen.input[0].previous_output.txid, Txid::all_zeros());
        assert_eq!(gen.input[0].previous_output.vout, 0xFFFFFFFF);
        // assert_eq!(serialize(&gen.input[0].script_sig),
        //            hex!("4d04ffff001d0104455468652054696d65732030332f4a616e2f32303039204368616e63656c6c6f72206f6e206272696e6b206f66207365636f6e64206261696c6f757420666f722062616e6b73"));
        // TODO: update hex value
        assert_eq!(gen.input[0].sequence, Sequence::MAX);
        assert_eq!(gen.output.len(), 1);
        // assert_eq!(serialize(&gen.output[0].script_pubkey),
        //            hex!("434104c10e83b2703ccf322f7dbd62dd5855ac7c10bd055814ce121ba32607d573b8810c02c0582aed05b4deb9c4b77b26d92428c61256cd42774babea0a073b2ed0c9ac"));
        // TODO: update hex value
        assert_eq!(gen.output[0].value, Amount::from_str("50 KAON").unwrap());
        assert_eq!(gen.lock_time, absolute::LockTime::ZERO);

        assert_eq!(
            gen.compute_wtxid().to_string(),
            "c28557d51efc033c0ec6c6065502a6b657c6182aee3b181ea90edd0729d770af"
        );
    }

    #[test]
    fn bitcoin_genesis_block_calling_convention() {
        // This is the best.
        let _ = genesis_block(&params::MAINNET);
        // this works and is ok too.
        let _ = genesis_block(&Network::Mainnet);
        let _ = genesis_block(Network::Mainnet);
        // This works too, but is suboptimal because it inlines the const.
        let _ = genesis_block(Params::MAINNET);
        let _ = genesis_block(&Params::MAINNET);
    }

    #[test]
    fn bitcoin_genesis_full_block() {
        let gen = genesis_block(&params::Mainnet);

        assert_eq!(gen.header.version, block::Version::ONE);
        assert_eq!(gen.header.prev_blockhash, BlockHash::all_zeros());
        assert_eq!(
            gen.header.merkle_root.to_string(),
            "b66cb243101fe049026c40b4aa1bd45a02aa36aedb75da8e466c8069efc28667"
        );

        assert_eq!(gen.header.time, 1638961350);
        assert_eq!(gen.header.bits, CompactTarget::from_consensus(0x1E0FFFF0));
        assert_eq!(gen.header.nonce, 607505);
        assert_eq!(
            gen.header.block_hash().to_string(),
            "82d0b6e4be278b3ad1aee7d2dc13253ee4022b06e5934831b4a0a70f1769fc4e"
        );
    }

    #[test]
    fn testnet_genesis_full_block() {
        let gen = genesis_block(&params::TESTNET);
        assert_eq!(gen.header.version, block::Version::ONE);
        assert_eq!(gen.header.prev_blockhash, BlockHash::all_zeros());
        assert_eq!(
            gen.header.merkle_root.to_string(),
            "c28557d51efc033c0ec6c6065502a6b657c6182aee3b181ea90edd0729d770af"
        );
        assert_eq!(gen.header.time, 1504695028);
        assert_eq!(gen.header.bits, CompactTarget::from_consensus(0x1f00ffff));
        assert_eq!(gen.header.nonce, 8026361);
        assert_eq!(
            gen.header.block_hash().to_string(),
            "656746ec382207f7e8cc77681cda67c742a1a16fa06a9301ba1f56aedbd85216"
        );
    }

    #[test]
    fn signet_genesis_full_block() {
        let gen = genesis_block(&params::SIGNET);
        assert_eq!(gen.header.version, block::Version::ONE);
        assert_eq!(gen.header.prev_blockhash, BlockHash::all_zeros());
        assert_eq!(
            gen.header.merkle_root.to_string(),
            "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
        );
        assert_eq!(gen.header.time, 1504695029);
        assert_eq!(gen.header.bits, CompactTarget::from_consensus(0x1f00ffff));
        assert_eq!(gen.header.nonce, 8026361);
        assert_eq!(
            gen.header.block_hash().to_string(),
            "00000008819873e925422c1ff0f99f7cc9bbb232af63a077a480a3633bee1ef6"
        );
    }

    // The *_chain_hash tests are sanity/regression tests, they verify that the const byte array
    // representing the genesis block is the same as that created by hashing the genesis block.
    fn chain_hash_and_genesis_block(network: Network) {
        use hashes::sha256;

        // The genesis block hash is a double-sha256 and it is displayed backwards.
        let genesis_hash = genesis_block(network).block_hash();
        // We abuse the sha256 hash here so we get a LowerHex impl that does not print the hex backwards.
        let hash = sha256::Hash::from_slice(genesis_hash.as_byte_array()).unwrap();
        let want = format!("{:02x}", hash);

        let chain_hash = ChainHash::using_genesis_block_const(network);
        let got = format!("{:02x}", chain_hash);

        // Compare strings because the spec specifically states how the chain hash must encode to hex.
        assert_eq!(got, want);

        #[allow(unreachable_patterns)] // This is specifically trying to catch later added variants.
        match network {
            Network::Mainnet => {},
            Network::Testnet => {},
            Network::Signet => {},
            Network::Regtest => {},
            _ => panic!("Update ChainHash::using_genesis_block and chain_hash_genesis_block with new variants"),
        }
    }

    macro_rules! chain_hash_genesis_block {
        ($($test_name:ident, $network:expr);* $(;)*) => {
            $(
                #[test]
                fn $test_name() {
                    chain_hash_and_genesis_block($network);
                }
            )*
        }
    }

    chain_hash_genesis_block! {
        mainnet_chain_hash_genesis_block, Network::Mainnet;
        testnet_chain_hash_genesis_block, Network::Testnet;
        signet_chain_hash_genesis_block, Network::Signet;
        regtest_chain_hash_genesis_block, Network::Regtest;
    }

    // Test vector taken from: https://github.com/lightning/bolts/blob/master/00-introduction.md
    #[test]
    fn mainnet_chain_hash_test_vector() {
        let got = ChainHash::using_genesis_block_const(Network::Mainnet).to_string();
        let want = "82d0b6e4be278b3ad1aee7d2dc13253ee4022b06e5934831b4a0a70f1769fc4e";
        assert_eq!(got, want);
    }
}
