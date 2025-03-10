"""
This file contains the prompts used for generating synthetic data for fine-tuning the BERT model.
"""

PROMPT_V1 = """
    I am generating synthetic OUTPUT to fine-tune
    my BERT model for detecting misinformation. The goal is to classify
    text based on its accuracy and prevent the spread of false information.

    Label definitions:
    - false: Completely untrue or fabricated information.
    - partially true: Contains some truth but is misleading or lacks important context.
    - mostly true: Largely accurate but may have minor inaccuracies or omissions.
    - true: Entirely accurate and factual information.

    Follow these steps to generate an accurate OUTPUT:

    To ensure the generated OUTPUT matches the LABEL, follow this step-by-step reasoning:
    1. Consider the CATEGORY and TYPE to understand the topic.
    2. Think about common statements or beliefs that align with the LABEL.
    3. Reflect on whether the statement should be false, partially true, mostly true, or true.
    4. Generate an OUTPUT that aligns with the assigned LABEL and category.

    Examples:

    LABEL: false
    CATEGORY: health and medicine
    TYPE: vaccines (efficacy, safety, side effects)
    OUTPUT: Vaccines contain microchips to track people.
    REASONING: This statement is false, because vaccines are developed to prevent diseases by triggering an immune response, not to track individuals. The ingredients in vaccines are publicly disclosed by manufacturers and are subject to rigorous testing and regulation by health authorities like FDA and WHO.

    LABEL: mostly true
    CATEGORY: climate change and environmental issues
    TYPE: conservation efforts or endangered species
    OUTPUT: Polar bear populations are declining due to melting Arctic ice.
    REASONING: This statement is mostly true because many polar bear populations are under threat as their sea ice habitat diminishes due to climate change. However, it may not be universally true for all polar bear populations, as some subpopulations have remained stable or even increased. The statement is largely accurate but may lack nuance regarding regional differences.

    LABEL: partially true
    CATEGORY: health and medicine
    TYPE: diet fads and health benefits
    OUTPUT: Eating carrots significantly improves your night vision.
    REASONING: This contains some truth because carrots are rich in beta-carotene, which is good for overall eye health, but the idea that they significantly improve night vision is misleading. The myth originated from British propaganda during World War II to explain their pilots' improved night-time operations.

    LABEL: true
    CATEGORY: health and medicine
    TYPE: COVID-19 (virus origins, treatments, lockdowns)
    OUTPUT: Wearing masks can help reduce the transmission of respiratory viruses.
    REASONING: This is true and factual information. Numerous studies have shown that masks are effective in reducing the spread of viruses, including the novel coronavirus.
    ######################################
    Your task is to generate one OUTPUT for the classification below. It's extremely important that the generated OUTPUT aligns with the assigned LABEL.
    Only return the OUTPUT and REASONING. Do not return the LABEL, CATEGORY, or TYPE.
    """

PROMPT_V2 = """
    I am generating synthetic OUTPUT to fine-tune my BERT model for detecting misinformation. 
    The goal is to classify text based on its accuracy and prevent the spread of false information.

    Label definitions:
    - false: Completely untrue or fabricated information.
    - partially true: Contains some truth but is misleading or lacks important context.
    - mostly true: Largely accurate but may have minor inaccuracies or omissions.
    - true: Entirely accurate and factual information.

    ### Step-by-Step Process:

    1. First, generate a factual and verifiable statement (TRUE OUTPUT) based on the CATEGORY and TYPE.
    2. Then, modify the TRUE OUTPUT according to the assigned LABEL:
    - false: Completely alter the statement to be misleading or fabricated.
    - partially true: Add misleading information or remove key context to make the statement somewhat deceptive.
    - mostly true: Slightly alter the statement to introduce minor inaccuracies while keeping most of it correct.
    - true: Keep the statement unchanged.
    3. Provide a REASONING to justify why the modified OUTPUT aligns with the assigned LABEL.

    ### Examples:

    CATEGORY: health and medicine  
    TYPE: vaccines (efficacy, safety, side effects)  
    TRUE OUTPUT: Vaccines protect against diseases by stimulating an immune response and have been proven safe through rigorous testing.  

    Modified Outputs:

    - LABEL: false
    OUTPUT: Vaccines are designed to alter human DNA permanently.  
    REASONING: This is false because vaccines work by training the immune system to recognize pathogens, not by altering DNA. No approved vaccines modify human genetic material.

    - LABEL: partially true
    OUTPUT: Vaccines can sometimes cause serious long-term health effects.  
    REASONING: This contains some truth because vaccines can have rare side effects, but they are extensively tested for safety. The statement is misleading as it overstates the risks.

    - LABEL: mostly true
    OUTPUT: Vaccines protect against most diseases, but they guarantee 100% immunity.  
    REASONING: This is mostly true because vaccines significantly reduce the risk of infection, but they do not provide absolute immunity in all cases.

    - LABEL: true
    OUTPUT: Vaccines protect against diseases by stimulating an immune response and have been proven safe through rigorous testing.  
    REASONING: This is entirely accurate, as vaccines undergo extensive trials and regulatory approvals before being distributed.

    ######################################
    Your task is to generate one OUTPUT for the classification below.  
    First, create a TRUE OUTPUT based on the CATEGORY and TYPE.  
    Then modify the OUTPUT according to the LABEL.  
    Only return the OUTPUT and REASONING. Do not return the TRUE OUTPUT, LABEL, CATEGORY, or TYPE.
"""
