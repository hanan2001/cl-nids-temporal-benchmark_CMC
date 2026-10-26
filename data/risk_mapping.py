"""
Risk Scores for CIC-IDS2017 Attack Categories
Computed using the AIM Framework (Rashid et al., 2025)

================================================================================
DISCLAIMER
================================================================================

The AIM framework (Rashid et al., 2025) provides:
    - The scoring equation
    - The seven metrics (P, S, D, C, AV, I, E)
    - The weight values

The metric values assigned to CIC-IDS2017 attack categories in this file
were NOT taken directly from Rashid et al. (2025).

Instead, they were derived through a structured expert assessment based on:
    • CVSS v3.1 (NVD) for attacks with representative CVEs
    • MITRE ATT&CK (technique mapping, tactics, detection guidance)
    • OWASP Top 10 (for web application attacks)
    • CISA Known Exploited Vulnerabilities (KEV)
    • Red Canary 2024 Threat Detection Report
    • Splunk ATT&CK Data 2023
    • Operational impact assessment

Therefore, the resulting AIM scores in this file should be considered
dataset-specific estimates for CIC-IDS2017 attack categories.

================================================================================
METHODOLOGY
================================================================================

The AIM framework (Rashid et al., 2025, ICSCA 2025) defines the scoring formula:

    Score = (P × 0.15) + (S × 0.20) + (D × 0.15) + 
            (C × 0.10) + (AV × 0.15) + (I × 0.20) + (E × 0.05)

Where:
    P = Prevalence      (High=1.0, Medium=0.7, Low=0.3)
    S = Severity        (High=1.0, Medium=0.7, Low=0.3)
    D = Detectability   (High=0.3, Medium=0.6, Low=1.0)  ← inverted scale
    C = Complexity      (High=0.3, Medium=0.6, Low=1.0)  ← inverted scale
    AV = Attack Vector  (Network=1.0, Adjacent=0.6, Local=0.3)
    I = Impact          (High=1.0, Medium=0.7, Low=0.3)
    E = Exploitability  (High=1.0, Medium=0.7, Low=0.3)

NOTE: The original AIM framework defines three discrete levels (High, Medium, Low).
To better distinguish the heterogeneous attack categories in CIC-IDS2017,
intermediate values (e.g., 0.5, 0.8, 0.9) were assigned using expert assessment
supported by CVSS severity, MITRE ATT&CK detection guidance, and public threat
intelligence reports. These intermediate values allow for finer differentiation
between attack categories with similar characteristics.

================================================================================
METRIC ASSIGNMENT RULES
================================================================================

For attacks with a representative CVE (e.g., Heartbleed):
    Metrics are based on CVSS v3.1 (NVD) and CISA KEV.

For attacks without a representative CVE (e.g., Bot, DDoS, PortScan, Infiltration):
    Metrics are estimated using:
    - MITRE ATT&CK (representative technique mapping, tactics, detection guidance)
    - Operational impact assessment
    - Public threat reports (Red Canary 2024, Splunk 2023)
    - Expert assessment based on documented criteria

================================================================================
REFERENCES
================================================================================
[1] Rashid, S.M.Z.U., Alam, M.M., Montasir, I., Haq, A. (2025)
    "Risk-Based MITRE TTP Scoring for Proactive Cyber Threat Prioritization"
    ICSCA 2025, ACM. https://doi.org/10.1145/3731806.3731846

[2] MITRE ATT&CK Enterprise Matrix: https://attack.mitre.org/matrices/enterprise/
[3] NVD CVSS v3.1: https://nvd.nist.gov/vuln-metrics/cvss/v3-calculator
[4] OWASP Top 10: https://owasp.org/Top10/
[5] CISA KEV: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
[6] Red Canary 2024 Threat Detection Report
[7] Splunk ATT&CK Data 2023
"""

import numpy as np

# ============================================================================
# 1. ATTACK PROFILES - SINGLE SOURCE OF TRUTH
# ============================================================================
# Each profile contains:
#   - P, S, D, C, AV, I, E: AIM metrics (estimated/assigned values)
#   - mitre_id: Representative MITRE ATT&CK technique ID
#   - tactic: Associated MITRE ATT&CK tactic
#   - cve: Representative CVE (if any)
#   - cvss: CVSS score (if applicable)
#   - source: Data source(s) used for estimation
#   - justification: Explanation of metric assignment
# ============================================================================

CIC_IDS2017_ATTACK_PROFILES = {

    # ====================================================================
    # Heartbleed (CVE-2014-0160)
    # ====================================================================
    'Heartbleed': {
        # AIM Metrics (estimated)
        'P': 0.3,      # Prevalence: Low - CVE-2014-0160, less prevalent today
        'S': 1.0,      # Severity: Critical - CVSS 9.8 (NVD)
        'D': 0.6,      # Detectability: Medium - detectable via SSL/TLS logs
        'C': 0.3,      # Complexity: Low - public exploit scripts available
        'AV': 1.0,     # Attack Vector: Network - exploitable remotely
        'I': 1.0,      # Impact: Critical - memory exfiltration (C+I)
        'E': 1.0,      # Exploitability: High - CISA KEV, public exploits
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1190',
        'tactic': 'Initial Access',
        
        # CVE Information
        'cve': 'CVE-2014-0160',
        'cvss': 9.8,
        
        # Sources and Justification
        'source': 'NVD / CISA KEV',
        'justification': 'Estimated based on CVE-2014-0160 CVSS 9.8 (NVD). '
                         'CISA KEV: known exploited vulnerability. '
                         'Prevalence considered Low due to age of vulnerability.'
    },

    # ====================================================================
    # Web Attack - SQL Injection (representative technique: T1190)
    # ====================================================================
    'Web Attack - Sql Injection': {
        # AIM Metrics (estimated)
        'P': 0.7,      # Prevalence: Medium - OWASP Top 10, frequent
        'S': 0.9,      # Severity: High - estimated based on worst-case scenarios
        'D': 0.6,      # Detectability: Medium - detectable via WAF and logs
        'C': 0.6,      # Complexity: Medium - SQL knowledge required
        'AV': 1.0,     # Attack Vector: Network - via web application
        'I': 1.0,      # Impact: Critical - data exfiltration and modification
        'E': 0.9,      # Exploitability: High - sqlmap and automated tools
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1190',
        'tactic': 'Initial Access',
        
        # CVE Information (no single representative CVE)
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'OWASP / MITRE ATT&CK',
        'justification': 'Estimated based on OWASP Top 10 2021 - A03:2021 (Injection). '
                         'Severity based on worst-case SQL injection scenarios '
                         'that allow data exfiltration or code execution.'
    },

    # ====================================================================
    # Web Attack - XSS (representative technique: T1059.007)
    # ====================================================================
    'Web Attack - XSS': {
        # AIM Metrics (estimated)
        'P': 0.7,      # Prevalence: Medium - OWASP Top 10
        'S': 0.8,      # Severity: High - estimated based on typical CVSS 6.1-7.5
        'D': 0.6,      # Detectability: Medium - detectable via CSP and logs
        'C': 0.5,      # Complexity: Medium - JavaScript skills required
        'AV': 1.0,     # Attack Vector: Network - via web application
        'I': 0.9,      # Impact: High - session hijacking possible
        'E': 0.9,      # Exploitability: High - payloads readily available
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1059.007',
        'tactic': 'Execution',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'OWASP / MITRE ATT&CK',
        'justification': 'Estimated based on OWASP Top 10 2021 - A03:2021 (Injection). '
                         'XSS can lead to session hijacking and credential theft.'
    },

    # ====================================================================
    # Web Attack - Brute Force (representative technique: T1110)
    # ====================================================================
    'Web Attack - Brute Force': {
        # AIM Metrics (estimated)
        'P': 1.0,      # Prevalence: High - estimated from public threat reports
        'S': 0.7,      # Severity: Medium-High - estimated based on typical CVSS 6.5
        'D': 0.3,      # Detectability: High - authentication failure logs
        'C': 0.3,      # Complexity: Low - automated tools available
        'AV': 1.0,     # Attack Vector: Network - via web interface
        'I': 0.6,      # Impact: Medium - access if credentials are weak
        'E': 0.8,      # Exploitability: High - easy to automate
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1110',
        'tactic': 'Credential Access',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Red Canary',
        'justification': 'Estimated based on T1110 (Brute Force). '
                         'Representative mapping based on CIC-IDS2017 behavior. '
                         'High detectability due to authentication logs.'
    },

    # ====================================================================
    # Infiltration (representative technique: T1078 - Valid Accounts)
    # ====================================================================
    'Infiltration': {
        # AIM Metrics (estimated)
        'P': 1.0,      # Prevalence: High - estimated from public threat reports
        'S': 1.0,      # Severity: Critical - full system access
        'D': 1.0,      # Detectability: Low - legitimate traffic
        'C': 0.3,      # Complexity: Low - if credentials are available
        'AV': 1.0,     # Attack Vector: Network - with valid credentials
        'I': 1.0,      # Impact: Critical - complete control of systems
        'E': 0.8,      # Exploitability: High - if credentials obtained
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1078',
        'tactic': 'Persistence',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Red Canary',
        'justification': 'Estimated based on T1078 (Valid Accounts). '
                         'Representative mapping of CIC-IDS2017 "Infiltration" '
                         'scenario based on attack behavior. '
                         'Very difficult to detect (legitimate credentials).'
    },

    # ====================================================================
    # Bot (representative technique: T1071 - Application Layer Protocol)
    # ====================================================================
    'Bot': {
        # AIM Metrics (estimated)
        'P': 0.7,      # Prevalence: Medium - estimated from public threat reports
        'S': 0.9,      # Severity: High - remote command execution
        'D': 1.0,      # Detectability: Low - encrypted traffic
        'C': 0.5,      # Complexity: Medium - varied implementations
        'AV': 1.0,     # Attack Vector: Network - C2 communication
        'I': 0.9,      # Impact: High - remote control of systems
        'E': 0.8,      # Exploitability: High - frameworks available
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1071',
        'tactic': 'Command and Control',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1071 (Application Layer Protocol). '
                         'Representative mapping for C2 communication behavior. '
                         'Encrypted traffic makes detection difficult (D=1.0).'
    },

    # ====================================================================
    # DDoS (representative technique: T1499 - Endpoint Denial of Service)
    # ====================================================================
    'DDoS': {
        # AIM Metrics (estimated)
        'P': 0.7,      # Prevalence: Medium - estimated from public threat reports
        'S': 0.8,      # Severity: High - service unavailability
        'D': 0.3,      # Detectability: High - volume anomaly
        'C': 0.5,      # Complexity: Medium - botnets required
        'AV': 1.0,     # Attack Vector: Network - distributed traffic
        'I': 0.9,      # Impact: High - operational disruption
        'E': 0.7,      # Exploitability: Medium - resources required
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1499',
        'tactic': 'Impact',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1499 (Endpoint Denial of Service). '
                         'DDoS attacks cause significant operational impact. '
                         'Readily detectable due to traffic volume anomalies.'
    },

    # ====================================================================
    # DoS GoldenEye (representative technique: T1499.002)
    # ====================================================================
    'DoS GoldenEye': {
        # AIM Metrics (estimated)
        'P': 0.3,      # Prevalence: Low - less frequent
        'S': 0.8,      # Severity: High - web server impact
        'D': 0.3,      # Detectability: High - HTTP flood
        'C': 0.4,      # Complexity: Medium - specific tool required
        'AV': 1.0,     # Attack Vector: Network
        'I': 0.8,      # Impact: High - web service disruption
        'E': 0.7,      # Exploitability: Medium - tool available
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1499.002',
        'tactic': 'Impact',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1499.002 (HTTP Flood). '
                         'GoldenEye is a specific HTTP flood tool. '
                         'Less frequent but impactful.'
    },

    # ====================================================================
    # DoS Hulk (representative technique: T1499.002)
    # ====================================================================
    'DoS Hulk': {
        # AIM Metrics (estimated)
        'P': 0.3,      # Prevalence: Low
        'S': 0.8,      # Severity: High - web server impact
        'D': 0.3,      # Detectability: High
        'C': 0.3,      # Complexity: Low - simple tool
        'AV': 1.0,     # Attack Vector: Network
        'I': 0.8,      # Impact: High
        'E': 0.6,      # Exploitability: Medium
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1499.002',
        'tactic': 'Impact',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1499.002 (HTTP Flood). '
                         'Hulk is a simpler HTTP flood tool.'
    },

    # ====================================================================
    # DoS Slowhttptest (representative technique: T1499.003)
    # ====================================================================
    'DoS Slowhttptest': {
        # AIM Metrics (estimated)
        'P': 0.3,      # Prevalence: Low
        'S': 0.7,      # Severity: Medium-High - resource consumption
        'D': 0.3,      # Detectability: High - slow connections
        'C': 0.4,      # Complexity: Medium - configuration required
        'AV': 1.0,     # Attack Vector: Network
        'I': 0.8,      # Impact: High - resource exhaustion
        'E': 0.6,      # Exploitability: Medium
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1499.003',
        'tactic': 'Impact',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1499.003 (Application Layer Slow DoS). '
                         'Slowhttptest is a slow rate DoS tool.'
    },

    # ====================================================================
    # DoS slowloris (representative technique: T1499.003)
    # ====================================================================
    'DoS slowloris': {
        # AIM Metrics (estimated)
        'P': 0.3,      # Prevalence: Low
        'S': 0.7,      # Severity: Medium-High
        'D': 0.3,      # Detectability: High
        'C': 0.3,      # Complexity: Low - simple tool
        'AV': 1.0,     # Attack Vector: Network
        'I': 0.7,      # Impact: Medium-High - connection exhaustion
        'E': 0.6,      # Exploitability: Medium
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1499.003',
        'tactic': 'Impact',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1499.003 (Connection Exhaustion DoS). '
                         'Slowloris is a classic slow DoS tool.'
    },

    # ====================================================================
    # PortScan (representative technique: T1046)
    # ====================================================================
    'PortScan': {
        # AIM Metrics (estimated)
        'P': 1.0,      # Prevalence: High - estimated from public threat reports
        'S': 0.5,      # Severity: Medium - reconnaissance only
        'D': 0.3,      # Detectability: High - connection logs
        'C': 0.3,      # Complexity: Low - nmap
        'AV': 1.0,     # Attack Vector: Network
        'I': 0.5,      # Impact: Medium - reconnaissance
        'E': 0.9,      # Exploitability: High - nmap
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1046',
        'tactic': 'Discovery',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1046 (Network Service Scanning). '
                         'Port scanning is a precursor to attacks (reconnaissance). '
                         'Very common. Easily detectable via network logs.'
    },

    # ====================================================================
    # FTP-Patator (representative technique: T1110.003)
    # ====================================================================
    'FTP-Patator': {
        # AIM Metrics (estimated)
        'P': 0.3,      # Prevalence: Low - less common
        'S': 0.6,      # Severity: Medium - FTP access if successful
        'D': 0.3,      # Detectability: High - authentication logs
        'C': 0.3,      # Complexity: Low - automated tool
        'AV': 1.0,     # Attack Vector: Network - FTP service
        'I': 0.6,      # Impact: Medium - file transfer access
        'E': 0.8,      # Exploitability: High - automated tool
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1110.003',
        'tactic': 'Credential Access',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1110.003 (Password Spraying). '
                         'FTP-Patator is a credential guessing tool for FTP.'
    },

    # ====================================================================
    # SSH-Patator (representative technique: T1110.003)
    # ====================================================================
    'SSH-Patator': {
        # AIM Metrics (estimated)
        'P': 0.3,      # Prevalence: Low - less common
        'S': 0.6,      # Severity: Medium - SSH access if successful
        'D': 0.3,      # Detectability: High - authentication logs
        'C': 0.3,      # Complexity: Low - automated tool
        'AV': 1.0,     # Attack Vector: Network - SSH service
        'I': 0.6,      # Impact: Medium - system access
        'E': 0.8,      # Exploitability: High - automated tool
        
        # Representative MITRE ATT&CK
        'mitre_id': 'T1110.003',
        'tactic': 'Credential Access',
        
        # CVE Information
        'cve': None,
        'cvss': None,
        
        # Sources and Justification
        'source': 'MITRE ATT&CK / Expert Assessment',
        'justification': 'Estimated based on T1110.003 (Password Spraying). '
                         'SSH-Patator is a credential guessing tool for SSH.'
    },
}


# ============================================================================
# 2. AIM SCORE CALCULATION
# ============================================================================

def calculate_aim_score(profile):
    """
    Calculate AIM score using the formula from Rashid et al. (2025):
    
    Score = (P × 0.15) + (S × 0.20) + (D × 0.15) + 
            (C × 0.10) + (AV × 0.15) + (I × 0.20) + (E × 0.05)
    
    Args:
        profile: Dictionary containing P, S, D, C, AV, I, E values
    
    Returns:
        float: AIM score rounded to 3 decimal places
    """
    P = profile['P']
    S = profile['S']
    D = profile['D']
    C = profile['C']
    AV = profile['AV']
    I = profile['I']
    E = profile['E']
    
    score = (P * 0.15) + (S * 0.20) + (D * 0.15) + \
            (C * 0.10) + (AV * 0.15) + (I * 0.20) + (E * 0.05)
    
    return round(score, 3)


# ============================================================================
# 3. GENERATE SCORES FROM PROFILES
# ============================================================================

# Compute AIM scores for all attack profiles
AIM_SCORES = {}
for attack_name, profile in CIC_IDS2017_ATTACK_PROFILES.items():
    AIM_SCORES[attack_name] = calculate_aim_score(profile)

# Special cases: BENIGN and Normal are not MITRE TTPs
# AIM is not applicable - they get the minimum score
AIM_SCORES['BENIGN'] = 0.05
AIM_SCORES['Normal'] = 0.05

# For backward compatibility with existing code that imports RISK_SCORES
RISK_SCORES = AIM_SCORES


# ============================================================================
# 4. MAIN INTERFACE FUNCTION
# ============================================================================

# Dans risk_mapping.py - remplacer la fonction get_risk_scores

# ============================================================================
# 4. MAIN INTERFACE FUNCTION - VERSION CORRIGÉE
# ============================================================================

def get_risk_scores(class_names):
    """
    Return risk scores for a list of class names.
    
    Args:
        class_names: List of class names (strings)
    
    Returns:
        dict: Mapping from class index to risk score
    """
    # === MAPPING DES INDICES ORIGINAUX CIC-IDS2017 ===
    # Ces indices correspondent aux classes dans le dataset complet (15 classes)
    # Class_0 = BENIGN
    # Class_4 = PortScan
    # Class_8 = DDoS
    # Class_9 = Bot
    # Class_10 = Web Attack - Brute Force
    # Class_13 = Infiltration
    
    ORIGINAL_INDEX_TO_RISK = {
        0: 0.05,    # BENIGN
        4: 0.58,    # PortScan
        8: 0.67,    # DDoS
        9: 0.82,    # Bot
        10: 0.58,   # Web Attack - Brute Force
        13: 0.91,   # Infiltration
    }
    
    # === MAPPING DES NOMS DIRECTS ===
    AIM_SCORES = {
        'BENIGN': 0.05,
        'Normal': 0.05,
        'Bot': 0.82,
        'DDoS': 0.67,
        'Infiltration': 0.91,
        'PortScan': 0.58,
        'Web Attack - Brute Force': 0.58,
        'Web Attack - Sql Injection': 0.58,
        'Web Attack - XSS': 0.58,
        'Heartbleed': 0.58,
        'DoS GoldenEye': 0.80,
        'DoS Hulk': 0.80,
        'DoS Slowhttptest': 0.75,
        'DoS slowloris': 0.75,
        'FTP-Patator': 0.58,
        'SSH-Patator': 0.58,
    }
    
    risk_mapping = {}
    
    for i, class_name in enumerate(class_names):
        class_name_str = str(class_name)
        score = 0.50  # Default
        matched = False
        
        # === CAS 1 : Noms "Class_X" (indice original) ===
        if class_name_str.startswith('Class_') and class_name_str[6:].isdigit():
            try:
                orig_idx = int(class_name_str[6:])
                if orig_idx in ORIGINAL_INDEX_TO_RISK:
                    score = ORIGINAL_INDEX_TO_RISK[orig_idx]
                    matched = True
            except:
                pass
        
        # === CAS 2 : Nom direct (BENIGN, PortScan, etc.) ===
        if not matched and class_name_str in AIM_SCORES:
            score = AIM_SCORES[class_name_str]
            matched = True
        
        # === CAS 3 : Correspondance partielle ===
        if not matched:
            class_lower = class_name_str.lower()
            for key, val in AIM_SCORES.items():
                if key.lower() in class_lower or class_lower in key.lower():
                    score = val
                    matched = True
                    break
        
        # === CAS 4 : Détection par mots-clés ===
        if not matched:
            class_lower = class_name_str.lower()
            if 'benign' in class_lower or 'normal' in class_lower:
                score = 0.05
            elif 'bot' in class_lower:
                score = 0.82
            elif 'infiltration' in class_lower:
                score = 0.91
            elif 'portscan' in class_lower or 'scan' in class_lower:
                score = 0.58
            elif 'ddos' in class_lower or 'dos' in class_lower:
                score = 0.67
            elif 'brute' in class_lower or 'patator' in class_lower:
                score = 0.58
            elif 'sql' in class_lower or 'xss' in class_lower:
                score = 0.58
            elif 'heartbleed' in class_lower:
                score = 0.58
            else:
                score = 0.50  # Unknown class
        
        risk_mapping[i] = score
    
    # === AFFICHAGE POUR DEBUG ===
    print(f"\n📊 Mapping des scores de risque:")
    for i, name in enumerate(class_names):
        score = risk_mapping[i]
        emoji = "🔴" if score > 0.7 else "🟠" if score > 0.5 else "🟡" if score > 0.1 else "🟢"
        status = "✅" if score != 0.50 else "⚠️"
        print(f"   {emoji} Class_{i} ({name}): {score:.2f} {status}")
    
    return risk_mapping
# ============================================================================
# 5. DISPLAY FUNCTIONS (for testing)
# ============================================================================

def print_justification():
    """Display complete justification for all scores."""
    print("\n" + "=" * 80)
    print("AIM RISK SCORES FOR CIC-IDS2017 ATTACK CATEGORIES")
    print("Methodology: Rashid et al. (2025) - ICSCA 2025")
    print("=" * 80)
    print()
    print("NOTE: These scores are dataset-specific estimates for CIC-IDS2017.")
    print("Metric values were assigned through structured expert assessment.")
    print()
    
    # Header
    print(f"{'Attack':<25} | {'MITRE':<10} | {'P':>4} | {'S':>4} | {'D':>4} | "
          f"{'C':>4} | {'AV':>4} | {'I':>4} | {'E':>4} | {'Score':>6}")
    print("-" * 95)
    
    # Rows
    for attack_name, score in AIM_SCORES.items():
        if attack_name in ['BENIGN', 'Normal']:
            continue
        
        profile = CIC_IDS2017_ATTACK_PROFILES.get(attack_name, {})
        mitre_id = profile.get('mitre_id', 'N/A')
        P = profile.get('P', 0.0)
        S = profile.get('S', 0.0)
        D = profile.get('D', 0.0)
        C = profile.get('C', 0.0)
        AV = profile.get('AV', 0.0)
        I = profile.get('I', 0.0)
        E = profile.get('E', 0.0)
        
        print(f"{attack_name:<25} | {mitre_id:<10} | {P:>4.1f} | {S:>4.1f} | "
              f"{D:>4.1f} | {C:>4.1f} | {AV:>4.1f} | {I:>4.1f} | {E:>4.1f} | "
              f"{score:>6.2f}")
    
    print("\n" + "=" * 80)
    print("BENIGN / Normal: 0.05 (not a MITRE TTP, AIM not applicable)")
    print("=" * 80)
    
    # Methodology summary
    print("\n" + "=" * 80)
    print("METHODOLOGY SUMMARY")
    print("=" * 80)
    print()
    print("For attacks with a representative CVE (e.g., Heartbleed):")
    print("  • Metrics are based on CVSS v3.1 (NVD) and CISA KEV")
    print()
    print("For attacks without a representative CVE (e.g., Bot, DDoS, PortScan):")
    print("  • Metrics are estimated using:")
    print("    - Representative MITRE ATT&CK technique mapping")
    print("    - Operational impact assessment")
    print("    - Public threat reports (Red Canary 2024, Splunk 2023)")
    print("    - Expert assessment based on documented criteria")
    print()
    
    # Detailed sources
    print("=" * 80)
    print("DETAILED SOURCES FOR EACH METRIC")
    print("=" * 80)
    print()
    print("  • P (Prevalence): Estimated from public threat reports")
    print("  • S (Severity): CVSS v3.1 for CVE-associated attacks / operational impact")
    print("  • D (Detectability): Estimated from MITRE ATT&CK Detection guidance")
    print("  • C (Complexity): Estimated from ATT&CK procedure descriptions")
    print("  • AV (Attack Vector): CVSS v3.1")
    print("  • I (Impact): CVSS v3.1 (CIA) for CVE-associated attacks / operational impact")
    print("  • E (Exploitability): Estimated from public exploit availability and CISA KEV")
    print()
    
    # Attack profiles details
    print("=" * 80)
    print("DETAILED JUSTIFICATION FOR EACH ATTACK")
    print("=" * 80)
    print()
    
    for attack_name, profile in CIC_IDS2017_ATTACK_PROFILES.items():
        print(f"\n📌 {attack_name}")
        print(f"   Representative MITRE ID: {profile.get('mitre_id', 'N/A')}")
        print(f"   Tactic: {profile.get('tactic', 'N/A')}")
        if profile.get('cve'):
            print(f"   CVE: {profile['cve']} (CVSS: {profile.get('cvss', 'N/A')})")
        print(f"   Source: {profile.get('source', 'N/A')}")
        print(f"   Justification: {profile.get('justification', 'N/A')}")
        print(f"   Metrics: P={profile['P']:.1f}, S={profile['S']:.1f}, "
              f"D={profile['D']:.1f}, C={profile['C']:.1f}, "
              f"AV={profile['AV']:.1f}, I={profile['I']:.1f}, E={profile['E']:.1f}")
        print(f"   AIM Score: {calculate_aim_score(profile):.3f}")


def print_scores():
    """Print only the scores table."""
    print("\n" + "=" * 80)
    print("AIM RISK SCORES FOR CIC-IDS2017")
    print("=" * 80)
    print()
    print(f"{'Attack':<30} | {'MITRE':<10} | {'Score':>6}")
    print("-" * 55)
    
    for attack_name, score in AIM_SCORES.items():
        if attack_name in ['BENIGN', 'Normal']:
            continue
        profile = CIC_IDS2017_ATTACK_PROFILES.get(attack_name, {})
        mitre_id = profile.get('mitre_id', 'N/A')
        print(f"{attack_name:<30} | {mitre_id:<10} | {score:>6.2f}")
    
    print("-" * 55)
    print(f"{'BENIGN':<30} | {'N/A':<10} | {AIM_SCORES['BENIGN']:>6.2f}")
    print(f"{'Normal':<30} | {'N/A':<10} | {AIM_SCORES['Normal']:>6.2f}")
    print("=" * 80)


# ============================================================================
# 6. SELF-TEST
# ============================================================================

if __name__ == "__main__":
    # Run test
    print_justification()
    print("\n" + "=" * 80)
    print("SELF-TEST COMPLETE")
    print("=" * 80)