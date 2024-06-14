# Security Policy

This security policy applies to the "core" crates in the rust-kaon ecosystem, which are
`kaon`, `secp256k1`, `kaon_hashes` and `kaon-internals`. These crates deal with
cryptography and cryptographic algorithms, and as such, are likely locations for security
vulnerabilities to crop up.

As a general rule, an issue is a security vulnerability if it could lead to:

* Loss of funds
* Loss of privacy
* Censorship (including e.g. by attaching an incorrectly low fee to a transaction)
* Any "ordinary" security problem, such as remote code execution or invalid memory access

In general, use your best judgement in determining whether an issue is a security issue. If not,
go ahead and post it to the public issue tracker.

**If you believe you are aware of a security issue**, please contact our team at
`info@kaon.one`.