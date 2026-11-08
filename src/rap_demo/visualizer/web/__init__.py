"""Web dashboard and API module for RAP Real-Time Visualizer."""

from rap_demo.visualizer.web.controller import SearchController, SearchStatus
from rap_demo.visualizer.web.app import create_app, app

__all__ = ["SearchController", "SearchStatus", "create_app", "app"]
