#!/bin/bash

set -e  # Exit on any error

# Get current version
CURRENT_VERSION=$(poetry version -s)
echo "Current version: $CURRENT_VERSION"

# Bump patch version using poetry
poetry version patch
NEW_VERSION=$(poetry version -s)
echo "New version: $NEW_VERSION"

# Update version in pyproject.toml [project] section as well
sed -i '' "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml

echo "Version bumped from $CURRENT_VERSION to $NEW_VERSION"

# Commit and push changes
git add -A && git commit -m "Bump version to $NEW_VERSION"
git push

echo "Build completed successfully!"
