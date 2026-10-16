"""Bind submission dependencies to the existing scheduler integration surface."""

from server.dependencies import DependencyProvider

# Reads stay live so monkeypatches/integrations on server.scheduler still work.
dependencies = DependencyProvider()
