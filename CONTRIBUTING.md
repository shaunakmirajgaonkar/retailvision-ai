# Contributing to RetailVision AI

Thanks for your interest in contributing! This project is an educational/engineering
exploration of local, cloud-free computer vision for retail analytics.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   pip3 install -r requirements.txt --break-system-packages
   ```
3. Set up PostgreSQL and run `schema.sql` (see `README.md`).
4. Create a branch for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Making Changes

- Keep modules focused — footfall counting, shelf monitoring, API, and dashboard
  are intentionally separate files.
- Follow existing code style (PEP 8, descriptive function/variable names).
- Add docstrings to new functions and classes.
- Test your changes locally before submitting (webcam or a sample video/image).

## Submitting a Pull Request

1. Commit your changes with a clear message describing what and why.
2. Push to your fork and open a pull request against `main`.
3. Describe what you changed, why, and how you tested it.
4. Be responsive to review feedback.

## Reporting Issues

Please include:
- Steps to reproduce
- Expected vs actual behavior
- Your OS, Python version, and relevant package versions
- Any error output/logs

## Code of Conduct

By participating, you agree to abide by the project's [Code of Conduct](CODE_OF_CONDUCT.md).
