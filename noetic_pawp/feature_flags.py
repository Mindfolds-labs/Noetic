from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class FeatureFlags:
    enable_wordspace: bool = False
    enable_ipa_channel: bool = False
    enable_associative_attention: bool = False
    enable_associative_memory: bool = False
    enable_rive_adapter: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


def add_feature_flag_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--enable-wordspace", action="store_true", default=FeatureFlags.enable_wordspace)
    parser.add_argument("--enable-ipa-channel", action="store_true", default=FeatureFlags.enable_ipa_channel)
    parser.add_argument(
        "--enable-associative-attention",
        action="store_true",
        default=FeatureFlags.enable_associative_attention,
    )
    parser.add_argument(
        "--enable-associative-memory",
        action="store_true",
        default=FeatureFlags.enable_associative_memory,
    )
    parser.add_argument("--enable-rive-adapter", action="store_true", default=FeatureFlags.enable_rive_adapter)


def feature_flags_from_args(args: Namespace) -> FeatureFlags:
    return FeatureFlags(
        enable_wordspace=bool(getattr(args, "enable_wordspace", False)),
        enable_ipa_channel=bool(getattr(args, "enable_ipa_channel", False)),
        enable_associative_attention=bool(getattr(args, "enable_associative_attention", False)),
        enable_associative_memory=bool(getattr(args, "enable_associative_memory", False)),
        enable_rive_adapter=bool(getattr(args, "enable_rive_adapter", False)),
    )
