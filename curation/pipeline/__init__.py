"""Pipeline stages used by the curation entry point.

The public names mirror the main flow: preprocess input rows, sample PRs, then
hydrate/process selected PRs. Lazy imports keep CLI help and settings parsing
fast because the heavy hydration modules are imported only when a run starts.
"""

__all__ = ["preprocess_prs", "sample_prs", "process_prs"]


def __getattr__(name: str):
    """Resolve public pipeline functions on first access."""
    if name == "process_prs":
        from curation.pipeline.hydration_pipeline import process_prs

        return process_prs
    if name == "preprocess_prs":
        from curation.pipeline.preprocessing_pipeline import preprocess_prs

        return preprocess_prs
    if name == "sample_prs":
        from curation.pipeline.sampler_pipeline import sample_prs

        return sample_prs
    raise AttributeError(name)
