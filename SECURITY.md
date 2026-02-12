# Security Advisory

## Recent Security Updates

This document tracks security updates made to the project dependencies.

### Update: February 2026

**Critical security vulnerabilities have been patched** by updating the following dependencies:

#### 1. MLflow: 2.7.1 → 3.5.0
**Reason**: Multiple critical vulnerabilities including:
- Path traversal vulnerabilities
- Remote code execution risks
- Unsafe deserialization issues
- DNS rebinding attacks
- Local file inclusion
- Cross-site scripting (XSS)
- Server-side request forgery (SSRF)
- Weak password requirements
- Insecure temporary file creation

**Patched in**: MLflow 3.5.0

#### 2. Apache Airflow: 2.7.1 → 2.10.1
**Reason**: Multiple security issues including:
- Proxy credentials leaking in task logs
- Execution with unnecessary privileges
- DAG author code execution in scheduler
- Permission verification bypass
- Pickle deserialization vulnerability in XComs
- Exposure of sensitive information

**Patched in**: Apache Airflow 2.10.1

#### 3. FastAPI: 0.103.1 → 0.109.1
**Reason**: 
- Content-Type Header ReDoS vulnerability

**Patched in**: FastAPI 0.109.1

#### 4. python-multipart: 0.0.6 → 0.0.22
**Reason**: Multiple vulnerabilities including:
- Arbitrary file write via non-default configuration
- Denial of Service (DoS) via malformed multipart/form-data boundary
- Content-Type Header ReDoS

**Patched in**: python-multipart 0.0.22

## Verification

To verify you're using the patched versions:

```bash
pip list | grep -E "(mlflow|apache-airflow|fastapi|python-multipart)"
```

Expected output:
```
apache-airflow           2.10.1
fastapi                  0.109.1
mlflow                   3.5.0
python-multipart         0.0.22
```

## Updating Your Installation

If you have an existing installation, update dependencies:

```bash
pip install -r requirements.txt --upgrade
```

Or run the setup script:

```bash
./scripts/setup_environment.sh
```

## Security Best Practices

To maintain security:

1. **Regular Updates**: Keep dependencies updated regularly
2. **Vulnerability Scanning**: Run `pip-audit` or similar tools periodically
3. **Environment Isolation**: Always use virtual environments
4. **Secrets Management**: Never commit credentials or API keys
5. **Network Security**: Use firewalls and restrict access to services
6. **Input Validation**: Leverage Pydantic for API input validation
7. **Logging**: Monitor logs for suspicious activity

## Reporting Security Issues

If you discover a security vulnerability in this project:

1. Do NOT open a public issue
2. Email the maintainers privately
3. Include detailed information about the vulnerability
4. Allow time for a fix before public disclosure

## Additional Resources

- [MLflow Security Documentation](https://mlflow.org/docs/latest/auth/index.html)
- [Airflow Security Documentation](https://airflow.apache.org/docs/apache-airflow/stable/security/index.html)
- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

## Changelog

- **2026-02-12**: Updated all vulnerable dependencies to patched versions
  - MLflow: 2.7.1 → 3.5.0
  - Apache Airflow: 2.7.1 → 2.10.1
  - FastAPI: 0.103.1 → 0.109.1
  - python-multipart: 0.0.6 → 0.0.22

---

**Status**: ✅ All known vulnerabilities patched

**Last Updated**: February 12, 2026
