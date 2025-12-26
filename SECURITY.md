# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in MetaGen, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- Acknowledgment within 48 hours
- Regular updates on progress
- Credit in the security advisory (if desired)

### Scope

MetaGen is a research artifact that generates code and documents. Security concerns include:

- **Code injection**: Malicious specs that generate harmful code
- **Path traversal**: Specs that write files outside intended directories
- **Denial of service**: Specs that cause excessive resource consumption

### Out of Scope

- Vulnerabilities in generated code (users should review before execution)
- Issues with third-party dependencies (report to upstream)
- Social engineering attacks

## Security Best Practices

When using MetaGen:

1. **Review generated code** before execution
2. **Validate specs** from untrusted sources
3. **Run in sandboxed environments** when testing untrusted specs
4. **Keep dependencies updated** (`pip install --upgrade`)

## Acknowledgments

We thank all security researchers who responsibly disclose vulnerabilities.
