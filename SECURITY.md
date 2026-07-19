# Security Policy

## Supported Versions

This is an educational/engineering project without formal version support.
Security fixes are applied to the `main` branch on a best-effort basis.

## Reporting a Vulnerability

If you discover a security vulnerability (e.g. SQL injection, credential
exposure, unsafe deserialization), please report it privately rather than
opening a public issue.

Contact: mirajgaonkarshaunak@gmail.com

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We'll acknowledge reports within a reasonable timeframe and work on a fix.

## Notes on This Project's Security Model

RetailVision AI is designed to run 100% locally:
- No data is sent to external/cloud services.
- Camera feeds, footfall events, and shelf data are stored only in your local
  PostgreSQL instance.
- Database credentials are read from environment variables — never commit a
  `.env` file with real credentials (see `.gitignore`).
- This project has not undergone a formal security audit and should not be
  deployed to production or exposed to the public internet without additional
  hardening (authentication on the API, input validation, TLS, etc.).
