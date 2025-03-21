"""
This file contains the labels and categories that will be used in the prompt.
"""

# Labels and Categories that will be used in the prompt
LABELS = ["false", "partially true", "mostly true", "true"]
CATEGORY_TYPE = {
    "health and medicine": [
        "COVID-19 (virus origins, treatments, lockdowns)",
        "vaccines (efficacy, safety, side effects)",
        "alternative medicine vs. scientific treatments",
        "diet fads and health benefits",
        "drug claims and cures",
    ],
    "politics and government": [
        "election integrity",
        "politicians or political events",
        "conspiracy theories around governmental institutions",
        "political narratives",
        "claims about political policies or actions",
    ],
    "climate change and environmental issues": [
        "facts about climate change",
        "renewable energy",
        "environmental disasters (e.g., wildfires, floods)",
        "conservation efforts or endangered species",
    ],
    "science and technology": [
        "flat Earth theories or other pseudoscientific claims",
        "space exploration",
        "AI fears",
        "5G technology",
    ],
    "conspiracy theories": [
        "QAnon or other large-scale conspiracy movements",
        "9/11 and false flag theories",
        "anti-science movements (e.g., anti-vaccine, anti-climate change)",
        "secret societies controlling world governments",
    ],
    "economics and financial markets": [
        "market crashes or investment schemes",
        "pyramid schemes and financial advice",
        "cryptocurrencies or blockchain",
        "economic collapse or manipulation",
    ],
    "social and cultural issues": [
        "racial, ethnic, or religious groups",
        "gender identity or LGBTQ+",
        "bias or selective reporting in media",
        "historical events",
    ],
    "technology and AI": [
        "AI or machine learning capabilities",
        "deepfake videos and synthetic content spread",
        "tech privacy breaches or hacking",
    ],
}
