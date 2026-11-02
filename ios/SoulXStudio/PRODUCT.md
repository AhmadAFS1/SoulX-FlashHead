# Product

<!-- impeccable:product-schema 1 -->

## Platform

ios

## Users

The repository owner testing SoulX FlashHead on an iPhone 12 Pro.

## Product Purpose

Select one of the two bundled repository portraits (Indian or Asian), initialize the PRO model locally, and stream generated talking-head video driven by selected audio on the phone.

## Capabilities and Constraints

Local Core ML inference, imported model packages, two predefined avatar cards, audio selection, initialization and generated-frame playback. Default to the Indian avatar. Model absence and errors must remain explicit. Real measured generation throughput must remain distinct from playback cadence. CUDA code remains independent.

## Stack

SwiftUI with Core ML and AVFoundation, selected as the native implementation for the explicitly requested iOS app.

## Product Principles

- Make the actual portrait and generated output the primary content.
- Surface initialization and buffering honestly.
- Keep all inference local after model installation.
