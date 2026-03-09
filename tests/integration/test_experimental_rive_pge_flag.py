from noetic_pawp.feature_flags import FeatureFlags


def test_rive_pge_stays_behind_experimental_flag() -> None:
    default_flags = FeatureFlags()
    experimental_flags = FeatureFlags(enable_experimental_rive_pge=True)

    assert default_flags.enable_experimental_rive_pge is False
    assert experimental_flags.enable_experimental_rive_pge is True
