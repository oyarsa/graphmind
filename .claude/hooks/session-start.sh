#!/bin/bash
set -euo pipefail

# Only run in Claude Code web environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "🚀 Setting up GraphMind development environment..."

# Install just if not already available
if ! command -v just &> /dev/null; then
  echo "📦 Installing just command runner..."
  cargo install just
  echo "✅ just installed successfully"
else
  echo "✅ just already installed"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies with uv..."
uv sync --extra cpu --extra baselines
echo "✅ Dependencies installed"

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
uv run pre-commit install
echo "✅ Pre-commit hooks installed"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "📝 Creating .env file from .env.example..."
  cp .env.example .env
  echo "✅ .env file created"
else
  echo "✅ .env file already exists"
fi

# Verify setup with just lint
echo "🔍 Verifying setup with just lint..."
just lint
echo "✅ Setup verification complete!"

echo "🎉 GraphMind development environment ready!"
