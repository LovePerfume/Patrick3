import anthropic
import fal_client
import os
import requests

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an expert architectural visualization prompt engineer specializing in photorealistic interior and exterior renders for hospitality spaces.

SOLSTICE is a dual-mode venue in Ho Chi Minh City, Vietnam:
- Ground floor of a neoclassical white building (Vinhomes Grand Park)
- 7m wide frontage, ~12.9m deep rectangular footprint
- DAY MODE: Modern specialty coffee lounge (warm, calm, sophisticated)
- NIGHT MODE: Cocktail lounge/bar (moody, atmospheric, vibrant neon)

KEY DESIGN ELEMENTS:
- Front terrace: Olive-sage powder-coated steel chairs, white/cream round tables, cream canvas parasols with brass poles, lush planters, terrazzo-effect pavers
- Interior banquette: Floor-to-ceiling walnut built-in shelves with books and ceramics, deep olive-green velvet banquette seating, warm brass reading lamps
- Cocktail bar counter: Black Marquina marble countertop with brass inlay edge, backlit amber resin shelving for spirits, antique brass tap fixtures
- Ceiling feature: RGBW LED strip cove — warm 2700K for day, neon green #00E5A0 for night mode
- Pendant lights: Smoked amber glass pendants with brass hardware above bar counter
- Lighting: Warm 2700K halogen for day; RGBW green + red neon-flex diagonals + pink ambient for night
- Materials: Walnut wood, Black Marquina marble, polished concrete floor, taupe linen upholstery, brushed brass accents, anthracite steel frames, Venetian plaster walls

Write Flux image generation prompts that are highly detailed, photorealistic, and technically accurate. Each prompt should be 80–120 words. Output ONLY the prompt text — no labels, no preamble, no explanation."""

SCENES = [
    {
        "num": 1,
        "title": "Front Terrace Exterior — Daytime",
        "desc": "Neoclassical white building facade with arched ground-floor openings. Front terrace with olive-sage powder-coated steel chairs and cream round tables, cream canvas parasols with brass poles. Lush tropical planters at the perimeter. Bright morning sunlight, warm and inviting. Street-level perspective, people having coffee outdoors."
    },
    {
        "num": 2,
        "title": "Interior Banquette + Bookshelf Wall — Daytime",
        "desc": "Floor-to-ceiling walnut built-in shelves filled with books, ceramics, and small plants. Deep olive-green velvet banquette seating along the full wall. Small marble-top tables, brass reading lamps. Warm 2700K lighting. Polished concrete floor. Circular woven rattan wall fixture. Venetian plaster walls. Cozy, café atmosphere. Soft natural light from front windows."
    },
    {
        "num": 3,
        "title": "Cocktail Bar Counter — Day/Evening",
        "desc": "Elegant bar counter with Black Marquina marble top and brass inlay edge. Backlit amber resin shelving displaying curated spirit bottles. Smoked amber glass pendant lights with brass hardware hanging above. Antique brass tap fixtures. Bar stools with taupe linen seats and anthracite steel frames. Warm directional lighting highlighting textures."
    },
    {
        "num": 4,
        "title": "Full Interior Overview — Daytime",
        "desc": "Wide-angle interior shot showing the full depth of the coffee lounge. Banquette bookshelf wall on the left, cocktail bar counter in the back-right. Scattered lounge chairs and marble-top café tables in the center. Warm 2700K ceiling cove lighting. Polished concrete floor. Venetian plaster walls. Natural light streaming through the front terrace openings. Calm morning atmosphere."
    },
    {
        "num": 5,
        "title": "Night Mode — Neon Ceiling Lounge",
        "desc": "Same interior transformed at night. Neon emerald green #00E5A0 LED strip cove lighting around the entire ceiling perimeter. Red neon-flex diagonal lines on the ceiling. Soft pink ambient glow from wall sconces. Dark moody atmosphere. People lounging with cocktails. Walnut shelves and green velvet banquette visible. Dramatic contrast between neon light and dark interior."
    },
    {
        "num": 6,
        "title": "Night Mode — Cocktail Bar Atmosphere",
        "desc": "Night-mode cocktail bar close-up. Smoked amber glass pendants dimmed to warm glow above Black Marquina marble bar counter. Green neon #00E5A0 reflecting off marble surface. Backlit amber spirit shelving glowing richly. Bartender silhouette. Red neon-flex accent lines on ceiling. Pink ambient light from background. Moody, intimate, sophisticated nightlife atmosphere."
    },
]

os.makedirs("renders", exist_ok=True)

print("SOLSTICE — Architectural Render Generator")
print("=" * 50)

for scene in SCENES:
    num = scene["num"]
    print(f"\n[{num}/6] Generating prompt for: {scene['title']}...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Write a Flux image generation prompt for this scene:\n\nScene: {scene['title']}\nDetails: {scene['desc']}\n\nRemember: output only the prompt text, 80–120 words, photorealistic architectural visualization style."
            }
        ]
    )

    flux_prompt = response.content[0].text.strip()
    cache_status = "CACHE HIT" if response.usage.cache_read_input_tokens > 0 else "cache miss"
    print(f"  Claude prompt generated ({cache_status})")
    print(f"  Prompt preview: {flux_prompt[:80]}...")

    print(f"  Rendering with fal.ai flux/dev...")
    result = fal_client.run(
        "fal-ai/flux/dev",
        arguments={
            "prompt": flux_prompt,
            "image_size": "landscape_16_9",
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "num_images": 1,
            "enable_safety_checker": True,
        }
    )

    image_url = result["images"][0]["url"]
    print(f"  Downloading render...")

    img_response = requests.get(image_url, timeout=60)
    img_response.raise_for_status()

    output_path = f"renders/render_{num:02d}.jpg"
    with open(output_path, "wb") as f:
        f.write(img_response.content)

    file_size_kb = os.path.getsize(output_path) // 1024
    print(f"  Saved: {output_path} ({file_size_kb} KB)")

print("\n" + "=" * 50)
print("All 6 renders complete. Files saved to renders/")
