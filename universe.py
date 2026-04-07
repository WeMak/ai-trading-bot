"""
Stock & Crypto Universe — Full Market Coverage
=================================================
~1900 US stocks (S&P 500 + S&P 400 MidCap + Russell 1000 + popular names)
150 crypto assets (top by market cap)

Used by: auto_trader.py, mass_trainer.py, sector_scanner.py
"""

# ─── SECTOR STOCKS (~1900 total) ────────────────────────────────
SECTORS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM", "ORCL", "ADBE", "CSCO", "INTC",
        "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "MU", "ADI",
        "NXPI", "FTNT", "PANW", "CRWD", "ZS", "NET", "DDOG", "SNOW", "PLTR", "HUBS",
        "TEAM", "WDAY", "NOW", "INTU", "ADSK", "ANSS", "KEYS", "MPWR", "ON", "SWKS",
        "MCHP", "TER", "ENTG", "SMCI", "SSNC", "EPAM", "GDDY", "PAYC", "IT", "MANH",
        "PTC", "WEX", "JNPR", "FFIV", "RPD", "VRNS", "QLYS", "TENB", "CYBR", "OKTA",
        "ESTC", "MDB", "PATH", "DOCN", "CFLT", "GTLB", "DT", "MNDY", "PCTY", "BILL",
        "APPF", "ENV", "NCNO", "ALTR", "SMAR", "BSY", "CWAN", "CSGP", "TYL", "TOST",
        "FROG", "BRZE", "APP", "ASAN", "RELY", "ZI", "RIOT", "MARA", "HUT", "BTBT",
        "GEN", "HPE", "HPQ", "DELL", "WDC", "STX", "NTAP", "PSTG", "SMTC", "ACLS",
        "MKSI", "FORM", "CEVA", "RMBS", "AMKR", "COHR", "LITE", "VIAV", "ITMR", "ONTO",
        "WOLF", "SLAB", "DIOD", "ALGM", "AMBA", "SITM", "PI", "AAOI", "OLED", "MVIS",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT", "SYK", "BDX", "BSX", "EW",
        "ZBH", "BAX", "HOLX", "ALGN", "DXCM", "PODD", "IDXX", "IQV", "CRL", "MTD",
        "A", "WAT", "TFX", "STE", "HSIC", "XRAY", "PDCO", "NRC", "LNTH", "RARE",
        "MRNA", "BNTX", "SGEN", "EXAS", "VEEV", "INCY", "ALNY", "SRRK", "PCVX", "RCKT",
        "BMRN", "NBIX", "HALO", "UTHR", "ARGX", "IONS", "SRPT", "PTCT", "FOLD", "APLS",
        "KRYS", "IMVT", "CRNX", "ARVN", "RVMD", "KRTX", "XENE", "BHVN", "PRTA", "DAWN",
        "HCA", "ELV", "CI", "CNC", "HUM", "MOH", "OSCR", "ALHC", "ACCD", "SDGR",
        "RMD", "SWAV", "AZTA", "BIO", "TECH", "NTRA", "GH", "TXG", "CRVL", "ENSG",
        "AMED", "PINC", "LMAT", "MMSI", "GKOS", "NVST", "OGN", "VTRS", "TAK", "ZTS",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB",
        "PNC", "TFC", "COF", "BK", "STT", "AIG", "MET", "PRU", "AFL", "TRV",
        "ALL", "PGR", "CB", "HIG", "FNF", "GL", "CINF", "RNR", "WRB", "EG",
        "MCO", "SPGI", "ICE", "NDAQ", "MSCI", "FDS", "MKTX", "CBOE", "CME", "VIRT",
        "V", "MA", "PYPL", "SQ", "FIS", "FISV", "GPN", "NCLH", "WU", "AFRM",
        "SOFI", "HOOD", "COIN", "LPLA", "RJF", "IBKR", "MKTX", "ALLY", "DFS", "SYF",
        "HBAN", "MTB", "CFG", "RF", "KEY", "FITB", "ZION", "CMA", "FHN", "SBNY",
        "EWBC", "WAL", "NYCB", "FRC", "PACW", "BOH", "OZK", "CUBI", "FFBC", "GBCI",
        "BRO", "MMC", "AON", "WTW", "AJG", "ERIE", "RYAN", "KNSL", "PLMR", "ROOT",
        "FAF", "FNF", "RLI", "SIGI", "KMPR", "THG", "ORI", "AFG", "HMN", "AMSF",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
        "MAR", "HLT", "RCL", "CCL", "ABNB", "EXPE", "LVS", "WYNN", "MGM", "CZR",
        "ROST", "TGT", "DG", "DLTR", "BBY", "WHR", "WSM", "ETSY", "W", "RH",
        "DECK", "CROX", "BIRK", "ON", "LULU", "GAP", "ANF", "AEO", "URBN", "RL",
        "TPR", "VFC", "PVH", "HBI", "CPRI", "SKX", "COLM", "UAA", "SHOO", "CRI",
        "DINE", "DRI", "EAT", "CAKE", "TXRH", "WING", "SHAK", "JACK", "DPZ", "YUM",
        "QSR", "WEN", "PZZA", "ARCO", "LOCO", "PLAY", "SIX", "FUN", "SEAS", "MTN",
        "DHI", "LEN", "NVR", "PHM", "TOL", "KBH", "MDC", "MHO", "GRBK", "TMHC",
        "CVNA", "KMX", "AZO", "ORLY", "AAP", "GPC", "LKQ", "MNRO", "PAG", "RACE",
        "F", "GM", "STLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "GOEV",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC",
        "GIS", "HSY", "K", "SJM", "MKC", "HRL", "TSN", "CAG", "CPB", "BG",
        "ADM", "STZ", "DEO", "BF-B", "SAM", "MNST", "CELH", "FIZZ", "COKE", "KDP",
        "EL", "CHD", "CLX", "KMB", "SPB", "HELE", "IPAR", "EPC", "ENR", "NUS",
        "SFM", "GO", "USFD", "PFGC", "CHEF", "HAIN", "LNCE", "POST", "THS", "SMPL",
        "CASY", "ARKO", "ACI", "KR", "BJ", "NGVC", "IMKTA", "VLGEA", "OLLI", "FIVE",
        "DNUT", "LANC", "FLO", "JBSS", "DAR", "INGR", "ANDE", "CALM", "VITL", "FRPT",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
        "DVN", "PXD", "FANG", "HES", "APA", "BKR", "TRGP", "OKE", "WMB", "KMI",
        "ET", "EPD", "MPLX", "PAA", "AM", "WES", "CTRA", "EQT", "RRC", "AR",
        "SWN", "MTDR", "PR", "CHRD", "SM", "ESTE", "NOG", "VTLE", "CIVI", "MGY",
        "HP", "RIG", "VAL", "NE", "OII", "WHD", "LBRT", "PUMP", "DEN", "PTEN",
        "DINO", "PBF", "DK", "CVI", "PARR", "HF", "RES", "CEIX", "BTU", "ARCH",
        "CNX", "AMPY", "REI", "CPE", "GPOR", "NEXT", "CHX", "FTI", "TDW", "CLB",
    ],
    "Industrials": [
        "CAT", "GE", "HON", "UPS", "BA", "RTX", "LMT", "DE", "UNP", "MMM",
        "NOC", "GD", "TXT", "HII", "LDOS", "BWXT", "KTOS", "MRCY", "HEI", "TDG",
        "FDX", "CSX", "NSC", "JBHT", "XPO", "SAIA", "ODFL", "KNX", "SNDR", "ARCB",
        "WM", "RSG", "WCN", "ECOL", "CLH", "GFL", "SRCL", "CWST", "ADSW", "HCCI",
        "EMR", "ETN", "ROK", "AME", "PH", "ITW", "SWK", "IR", "DOV", "GNRC",
        "FAST", "GWW", "MSM", "WSO", "AIT", "HDS", "SITE", "POOL", "SHW", "BLDR",
        "VMC", "MLM", "EXP", "USCR", "SMID", "TREX", "AZEK", "DOOR", "JELD", "AWI",
        "DAL", "UAL", "LUV", "ALK", "SAVE", "HA", "SKYW", "MESA", "ATSG", "AAWW",
        "WAB", "TT", "XYL", "REXR", "PCAR", "AGCO", "CNHI", "OSK", "ASTE", "WMS",
        "PWR", "TTEK", "ACM", "DY", "MTZ", "FIX", "MYRG", "PRIM", "BLD", "IEA",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "DOW", "NUE", "VMC",
        "PPG", "IFF", "CE", "ALB", "EMN", "HUN", "RPM", "AXTA", "CBT", "KWR",
        "STLD", "RS", "CMC", "ATI", "CRS", "ZEUS", "HAYN", "KALU", "CENX", "AA",
        "X", "CLF", "MT", "TX", "GGB", "PKG", "IP", "WRK", "SEE", "SON",
        "BCC", "MERC", "UFPI", "AWR", "OLN", "VNTR", "IOSP", "FUL", "TROX", "CC",
        "LTHM", "LAC", "SQM", "MP", "PLL", "VALE", "RIO", "BHP", "GOLD", "AEM",
        "FNV", "WPM", "RGLD", "OR", "AG", "HL", "CDE", "MAG", "PAAS", "SSRM",
    ],
    "Real Estate": [
        "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB",
        "ESS", "MAA", "UDR", "CPT", "INVH", "AMH", "REXR", "EGP", "FR", "STAG",
        "ARE", "OHI", "SBRA", "MPW", "PEAK", "VTR", "HR", "DOC", "CTRE", "LTC",
        "WPC", "NNN", "ADC", "EPRT", "STOR", "SRC", "BNL", "FCPT", "PINE", "GOOD",
        "CUBE", "EXR", "LSI", "NSA", "REXR", "COLD", "IIPR", "VICI", "GLPI", "RHP",
        "IRM", "SBAC", "UNIT", "LAUR", "GTY", "AKR", "BRX", "KRG", "RPT", "ROIC",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ES",
        "ED", "AWK", "ATO", "NI", "CMS", "DTE", "PPL", "FE", "ETR", "CEG",
        "VST", "PEG", "EVRG", "LNT", "BKH", "OGE", "PNW", "IDA", "AVA", "NWE",
        "ALE", "SWX", "UTL", "OTTR", "MSEX", "YORW", "SJW", "WTRG", "CWT", "AWR",
        "PCG", "EIX", "AES", "NRG", "ORA", "AQN", "BEP", "CWEN", "RNW", "NOVA",
    ],
    "Communication Services": [
        "META", "GOOGL", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA",
        "TTWO", "RBLX", "U", "SE", "BIDU", "TME", "SPOT", "ROKU", "PARA", "WBD",
        "LYV", "IMAX", "CNK", "MSGS", "FOXA", "NWSA", "NYT", "LEG", "IPG", "OMC",
        "WPP", "MGNI", "PUBM", "DSP", "TTD", "DV", "CARG", "ZD", "IAC", "ANGI",
        "MTCH", "BMBL", "SNAP", "PINS", "TWTR", "RDDT", "YELP", "GRPN", "CURI", "GENI",
        "SIRI", "LSXMA", "FWONA", "TKO", "EDR", "DASH", "GRAB", "CPNG", "MELI", "SE",
    ],
    "Biotech": [
        "BIIB", "ILMN", "DXCM", "BIO", "TECH", "MRNA", "BNTX", "NVAX", "INO", "VXRT",
        "CRISPR", "NTLA", "BEAM", "EDIT", "CRSP", "VERV", "PRME", "VIR", "CDMO", "DNA",
        "TWST", "PACB", "FATE", "NKTR", "SAGE", "SANA", "CERT", "RCKT", "BLUE", "LENZ",
        "RXRX", "RLAY", "ABCL", "ACAD", "ADVM", "AGIO", "ALEC", "ALLK", "AMTI", "ANNX",
    ],
    "Semiconductors": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC",
        "MRVL", "ADI", "ON", "NXPI", "MCHP", "SWKS", "MPWR", "ENTG", "TER", "RMBS",
        "COHR", "AMKR", "FORM", "CEVA", "SLAB", "DIOD", "AMBA", "SITM", "ALGM", "SMTC",
        "ARM", "TSM", "ASML", "UMC", "GFS", "WOLF", "ACLS", "MKSI", "ONTO", "IPGP",
    ],
    "Software & Cloud": [
        "CRM", "ORCL", "ADBE", "NOW", "INTU", "WDAY", "SNOW", "PLTR", "DDOG", "CRWD",
        "ZS", "NET", "HUBS", "TEAM", "PANW", "FTNT", "OKTA", "MDB", "ESTC", "CFLT",
        "GTLB", "DT", "MNDY", "BILL", "PCTY", "PAYC", "TOST", "FROG", "BRZE", "APP",
        "ASAN", "ZI", "PATH", "DOCN", "S", "TENB", "VRNS", "RPD", "CYBR", "QLYS",
    ],
    "AI & Robotics": [
        "NVDA", "MSFT", "GOOGL", "META", "AMZN", "PLTR", "AI", "BBAI", "SOUN", "PRCT",
        "ISRG", "IRBT", "TER", "BRKS", "ONTO", "PATH", "UPST", "LMND", "RKLB", "ASTS",
        "IONQ", "QUBT", "RGTI", "ARQQ", "QBTS", "CIFR", "CLSK", "WULF", "BTDR", "IREN",
    ],
    "EVs & Clean Energy": [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "GOEV", "MULN", "WKHS",
        "CHPT", "BLNK", "EVGO", "VLTA", "ENPH", "SEDG", "FSLR", "RUN", "NOVA", "ARRY",
        "PLUG", "BLDP", "BE", "FCEL", "HYZON", "SHLS", "FLNC", "STEM", "QS", "MVST",
    ],
}

# ─── EXTENDED UNIVERSE — S&P MidCap 400 + Russell extras ────────
EXTENDED = [
    # Mid-cap growth
    "DOCS", "GMED", "NUVB", "MGNI", "IRTC", "AZEK", "TREX", "RGEN", "CYTK", "LNTH",
    "CORT", "AXON", "TKO", "DUOL", "CELH", "ELF", "WFRD", "RBC", "THC", "ACHC",
    "SEM", "SGRY", "USPH", "AGIO", "RCKT", "VCEL", "RPRX", "MEDP", "IOVA", "RVNC",
    # Mid-cap value
    "OGN", "VTRS", "PRGO", "ELAN", "CTLT", "BHC", "HLF", "ATGE", "NSIT", "AYI",
    "JBSS", "FHI", "SPSC", "CW", "ENS", "EBC", "CATY", "FFIN", "SBCF", "FIBK",
    "PPBI", "WSFS", "HWC", "BUSE", "NWBI", "UBSI", "WAFD", "SFBS", "PNFP", "FULT",
    # Small-cap momentum
    "SATS", "SPWR", "MAXN", "ARRY", "BEEM", "ORA", "AY", "NEP", "CWEN", "RNW",
    "ARIS", "WTTR", "SWI", "JAMF", "VRNT", "PRFT", "BSIG", "STEP", "VCTR", "KW",
    "AIR", "KLIC", "CGNX", "NOVT", "ROCK", "LFUS", "CXT", "AZTA", "RVTY", "LECO",
    # Industrial mid-caps
    "ACA", "CBZ", "EXPO", "CRAI", "HURN", "FCN", "FORR", "ICFI", "KFRC", "MMS",
    "RHI", "KELYA", "HEIDRICK", "NSSC", "SXI", "RRX", "GATX", "AIT", "DORM", "NPO",
    "AIMC", "BWXT", "DCI", "FELE", "GTES", "HI", "MIDD", "MWA", "NDSN", "NN",
    # Financial mid-caps
    "VOYA", "LNC", "FAF", "ESNT", "MTG", "RDN", "NMIH", "AGO", "ESGR", "RYAN",
    "KNSL", "PLMR", "AMSF", "HMN", "SIGI", "THG", "AFG", "ORI", "KMPR", "WR",
    "TRMK", "GSBC", "BY", "RNST", "TOWN", "AUB", "SSB", "ONB", "FCNCA", "FRME",
    # Consumer mid-caps
    "BYD", "PENN", "DKNG", "RSI", "FLUT", "GDEN", "RRR", "CHDN", "MCRI", "BYD",
    "MOD", "PBH", "CENTA", "PRPL", "SNBR", "ETD", "FOXF", "THRM", "LCII", "WGO",
    "CWH", "PATK", "BC", "VC", "SITE", "POOL", "FBIN", "MAS", "FBHS", "AWI",
    # Tech mid-caps
    "FOUR", "EVTC", "RELY", "PAY", "RVLV", "BIGC", "VTEX", "PYCR", "ALKT", "CNXC",
    "TTEC", "TASK", "WNS", "GLOB", "EXLS", "EPAM", "CLVT", "RNG", "TWLO", "ZEN",
    "FIVN", "LPSN", "BAND", "EVCM", "RIOT", "MARA", "HUT", "BTBT", "BITF", "CIFR",
    # Healthcare mid-caps
    "CERT", "CORT", "SAVA", "PRTA", "ANNX", "KURA", "TGTX", "RAPT", "ARQT", "VRNA",
    "MGTA", "TALK", "MNKD", "SUPN", "COLL", "PCRX", "AMPH", "PAHC", "TARO", "PRGO",
    "ACAD", "SAGE", "AXSM", "CALA", "SGMO", "EDIT", "BEAM", "NTLA", "CRSP", "VERV",
    # Energy mid-caps
    "GPOR", "NEXT", "CHX", "FTI", "TDW", "CLB", "AROC", "USAC", "CCLP", "GEL",
    "TGS", "PBT", "UNT", "MUR", "EPSN", "TRP", "ENB", "AMLP", "ENLC", "DCP",
    "EGY", "TPVG", "FLMN", "CLMT", "CAPL", "SBOW", "REI", "CPE", "CDEV", "HNST",
    # Retail & restaurants
    "CAVA", "BROS", "DUTCH", "JACK", "KRUS", "NATH", "RRGB", "ARCO", "FWRG", "LOCO",
    "RUTH", "TACO", "EAT", "DINE", "BKC", "BJRI", "BLMN", "DIN", "CBRL", "DENN",
    "PZZA", "WEN", "LULU", "GIII", "SCVL", "BOOT", "HIBB", "DKS", "PLCE", "BURL",
    # Real estate mid-caps
    "AIRC", "APLE", "BRT", "CHCT", "CSR", "DEI", "ELME", "GIPR", "HIW", "KRC",
    "LXP", "MAC", "OFC", "PGRE", "SHO", "SKT", "STRA", "TCO", "VRE", "XHR",
    # Materials mid-caps
    "ARNC", "APEX", "BERY", "CBT", "CMP", "CSWI", "ECVT", "GEF", "GMS", "HCC",
    "KNF", "KRA", "MERC", "MTX", "OI", "PHIN", "SXT", "TRS", "USLM", "WOR",
    # Communication mid-caps
    "IHRT", "CMLS", "SALM", "NXST", "SSP", "GTN", "TRI", "SSP", "SCHL", "EDR",
    "CRI", "PLNT", "PTON", "BODY", "GENI", "DKNG", "RSI", "BALY", "AGS", "ACEL",
    # Utilities mid-caps
    "ALE", "SWX", "UTL", "OTTR", "MSEX", "YORW", "SJW", "WTRG", "CWT", "ARTNA",
    "AMPS", "NWN", "SPKE", "CWEN", "OGS", "SR", "POR", "BKH", "MGE", "MGEE",
    # Miscellaneous popular tickers
    "GME", "AMC", "BBBY", "BB", "NOK", "WISH", "CLOV", "SOFI", "OPEN", "RDFN",
    "Z", "ZG", "REAL", "COMP", "HIMS", "MNDY", "GRAB", "SE", "CPNG", "MELI",
    "NU", "STNE", "PAGS", "VTEX", "GLOB", "DLO", "BILL", "WIX", "SHOP", "MKTX",
    "TTD", "ROKU", "DOCU", "ZM", "OKTA", "CRWD", "NET", "DDOG", "SNOW", "MDB",
    # More large caps not in sectors
    "BRK-B", "JNJ", "V", "MA", "UNH", "XOM", "PG", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "WMT", "MCD", "CSCO",
    "ACN", "DHR", "ABT", "NKE", "TXN", "PM", "UPS", "MS", "BLK", "SCHW",
    "SPGI", "LOW", "DE", "BA", "ADP", "GILD", "SYK", "MDT", "ISRG", "BKNG",
    "ADI", "ZTS", "VRTX", "REGN", "TGT", "MMC", "CB", "PLD", "SO", "DUK",
    "ICE", "SHW", "CL", "CME", "CI", "MDLZ", "APD", "NOC", "TJX", "ORLY",
    "NSC", "FDX", "AON", "HUM", "SRE", "D", "AEP", "PGR", "TRV", "ALL",
    "AIG", "AFL", "MET", "PRU", "RE", "WRB", "L", "CINF", "BEN", "IVZ",
    # S&P 500 completions
    "AMCR", "AMP", "APTV", "BALL", "BG", "BIO", "BR", "BWA", "CAH", "CARR",
    "CBOE", "CDNS", "CE", "CF", "CHD", "CHRW", "CMA", "CPT", "CTRA", "CZR",
    "DAL", "DGX", "DPZ", "DRI", "DVA", "DXC", "EBAY", "ELAN", "EMN", "ENPH",
    "EPAM", "EQIX", "EQR", "EQT", "ESS", "EVRG", "EXR", "FBHS", "FE", "FIS",
    "FLT", "FOXA", "FRC", "FTNT", "GEN", "GL", "GPC", "GRMN", "GWW", "HAL",
    "HAS", "HES", "HIG", "HOLX", "HPE", "HPQ", "HST", "HSY", "HWM", "IAC",
    "IDXX", "IEX", "ILMN", "INCY", "IP", "IPG", "IQV", "IRM", "IT", "JBHT",
    "JKHY", "JCI", "JNPR", "KDP", "KEY", "KEYS", "KIM", "KLAC", "KMX", "KR",
    "L", "LDOS", "LH", "LKQ", "LNT", "LVS", "LW", "LYB", "LYV", "MAA",
    "MAS", "MKC", "MKTX", "MLM", "MOH", "MOS", "MRO", "MTCH", "MTD", "NDAQ",
    "NI", "NRG", "NTRS", "NVR", "NWSA", "ODFL", "OMC", "OKE", "OTIS", "PAYC",
    "PEAK", "PFG", "PKG", "PKI", "PNR", "PNW", "PPG", "PPL", "PTC", "PVH",
    "PWR", "RE", "REG", "RF", "RJF", "RMD", "ROL", "ROP", "ROST", "SBAC",
    "SEDG", "SEE", "SIVB", "SJM", "SNA", "SNPS", "SON", "SPG", "STE", "STX",
    "STZ", "SWK", "TAP", "TDY", "TECH", "TEL", "TER", "TFX", "TRMB", "TROW",
    "TXT", "UBER", "UDR", "ULTA", "URI", "VFC", "VICI", "VLO", "VMC", "VNO",
    "VRSK", "VTR", "VTRS", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WHR",
    "WMB", "WRK", "WST", "WTW", "WY", "WYNN", "XEL", "XYL", "YUM", "ZBH", "ZBRA",
    # Russell 2000 high-volume
    "ACHR", "AEHR", "ALAR", "ALGT", "AMSC", "APLD", "ARHS", "AVDL", "BLDE", "BOWL",
    "BTMD", "BYON", "CALM", "CARS", "CENTA", "CG", "CMPO", "CNMD", "COOP", "CRDO",
    "CVGW", "DIOD", "DOMO", "DV", "EGAN", "ENTA", "EVBG", "EVTL", "FCFS", "FLGT",
    "FN", "FRSH", "FSTR", "GATO", "GEOS", "GSHD", "HAYW", "HBI", "HLIT", "HNST",
    "HROW", "HSKA", "HZO", "ICUI", "IIPR", "IMXI", "INDB", "INGN", "INSM", "INTA",
    "IO", "IRDM", "ISTR", "ITCI", "KALU", "KFY", "KN", "KRNT", "LBRT", "LGIH",
    "LIVN", "LQDT", "LSTR", "MATX", "MBIN", "MCBS", "MGPI", "MMSI", "MOD", "MPAA",
    "MRCY", "MSTR", "MTSI", "NEOG", "NGVT", "NMIH", "NOVT", "NUVB", "OFIX", "OPCH",
    "OSCR", "OZK", "PAYO", "PECO", "PGNY", "PI", "PIPR", "PRGS", "PRLB", "PTGX",
    "PWSC", "QLYS", "RAMP", "RCKT", "RDW", "REVG", "REZI", "RMBS", "RNST", "SAH",
    "SANM", "SATS", "SBRA", "SCSC", "SHOO", "SITC", "SKWD", "SMPL", "SNEX", "SONO",
    "SPB", "SPNT", "SPSC", "STEP", "SWI", "TBBK", "TDC", "TGLS", "TMDX", "TNET",
    "TRMK", "TRUP", "TTMI", "TVTX", "UDMY", "UHAL", "USPH", "VCYT", "VECO", "VERX",
    "VIRT", "VREX", "VSEC", "WERN", "WK", "WOLF", "WTS", "XNCR", "XPRO", "YEXT",
    # International ADRs popular in US
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "GRAB", "SE", "CPNG",
    "MELI", "NU", "STNE", "PAGS", "VTEX", "GLOB", "DLO", "TSM", "ASML", "SAP",
    "TM", "HMC", "SNE", "SONY", "NVS", "AZN", "GSK", "BP", "SHEL", "TTE",
    "HSBC", "UBS", "DB", "CS", "BCS", "ING", "SAN", "BBVA", "RIO", "BHP",
    "VALE", "GOLD", "AEM", "FNV", "WPM", "TECK", "SCCO", "SQM", "LAC", "MP",
    # SPACs that became real companies
    "LCID", "JOBY", "LILM", "ACHR", "EVTL", "SPCE", "ASTR", "RKLB", "ASTS", "IRDM",
    "DNA", "GENI", "HIMS", "BARK", "BIRD", "OUST", "MAPS", "VIEW", "MVST", "QS",
    # ETFs to track for sentiment
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK", "ARKG", "ARKF", "ARKW",
    "XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC",
    "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "LQD", "JNK", "VNQ", "KWEB",
    "EEM", "EFA", "FXI", "INDA", "EWZ", "MCHI", "VWO", "IEMG", "SOXL", "TQQQ",
]

ALL_STOCKS = []
_seen = set()
for _tickers in SECTORS.values():
    for _t in _tickers:
        if _t not in _seen:
            ALL_STOCKS.append(_t)
            _seen.add(_t)
# Add extended universe
for _t in EXTENDED:
    if _t not in _seen:
        ALL_STOCKS.append(_t)
        _seen.add(_t)


# ─── CRYPTO UNIVERSE (150) ──────────────────────────────────────
CRYPTO_TICKERS = {
    # Tier 1 — Top 20 by Market Cap
    "BTC": "BTC-USD", "ETH": "ETH-USD", "BNB": "BNB-USD", "XRP": "XRP-USD",
    "ADA": "ADA-USD", "SOL": "SOL-USD", "DOGE": "DOGE-USD", "DOT": "DOT-USD",
    "AVAX": "AVAX-USD", "SHIB": "SHIB-USD", "MATIC": "MATIC-USD", "TRX": "TRX-USD",
    "LINK": "LINK-USD", "UNI": "UNI-USD", "ATOM": "ATOM-USD", "LTC": "LTC-USD",
    "BCH": "BCH-USD", "NEAR": "NEAR-USD", "APT": "APT-USD", "FIL": "FIL-USD",
    # Tier 2 — Top 50
    "ICP": "ICP-USD", "HBAR": "HBAR-USD", "VET": "VET-USD", "ALGO": "ALGO-USD",
    "QNT": "QNT-USD", "GRT": "GRT-USD", "FTM": "FTM-USD", "SAND": "SAND-USD",
    "MANA": "MANA-USD", "AAVE": "AAVE-USD", "XTZ": "XTZ-USD", "EOS": "EOS-USD",
    "THETA": "THETA-USD", "AXS": "AXS-USD", "IMX": "IMX-USD", "CRV": "CRV-USD",
    "SNX": "SNX-USD", "COMP": "COMP-USD", "MKR": "MKR-USD", "RPL": "RPL-USD",
    "LDO": "LDO-USD", "OP": "OP-USD", "ARB": "ARB-USD", "SUI": "SUI-USD",
    "SEI": "SEI-USD", "TIA": "TIA-USD", "INJ": "INJ-USD", "RUNE": "RUNE-USD",
    "STX": "STX-USD", "RNDR": "RNDR-USD",
    # Tier 3 — DeFi & Layer 2
    "SUSHI": "SUSHI-USD", "YFI": "YFI-USD", "BAL": "BAL-USD", "1INCH": "1INCH-USD",
    "DYDX": "DYDX-USD", "GMX": "GMX-USD", "CAKE": "CAKE-USD", "JOE": "JOE-USD",
    "SPELL": "SPELL-USD", "LQTY": "LQTY-USD", "PENDLE": "PENDLE-USD",
    "EIGEN": "EIGEN-USD", "STRK": "STRK-USD", "ZK": "ZK-USD",
    # Tier 4 — Gaming & Metaverse
    "GALA": "GALA-USD", "ENJ": "ENJ-USD", "ILV": "ILV-USD", "MAGIC": "MAGIC-USD",
    "YGG": "YGG-USD", "PYR": "PYR-USD", "WAXP": "WAXP-USD", "PRIME": "PRIME-USD",
    "BLUR": "BLUR-USD", "LOOKS": "LOOKS-USD",
    # Tier 5 — AI Crypto
    "FET": "FET-USD", "AGIX": "AGIX-USD", "OCEAN": "OCEAN-USD", "NMR": "NMR-USD",
    "TAO": "TAO-USD", "AKT": "AKT-USD", "OLAS": "OLAS-USD", "AIOZ": "AIOZ-USD",
    "CTXC": "CTXC-USD", "ROSE": "ROSE-USD",
    # Tier 6 — Memecoins & Trending
    "PEPE": "PEPE-USD", "FLOKI": "FLOKI-USD", "WIF": "WIF-USD", "BONK": "BONK-USD",
    "MEME": "MEME-USD", "BABYDOGE": "BABYDOGE-USD", "TURBO": "TURBO-USD",
    "NEIRO": "NEIRO-USD", "POPCAT": "POPCAT-USD", "MOG": "MOG-USD",
    # Tier 7 — Infrastructure & Storage
    "FIL": "FIL-USD", "AR": "AR-USD", "SC": "SC-USD", "STORJ": "STORJ-USD",
    "ANKR": "ANKR-USD", "CELO": "CELO-USD", "KAVA": "KAVA-USD", "ONE": "ONE-USD",
    "ZIL": "ZIL-USD", "IOTA": "IOTA-USD", "XLM": "XLM-USD",
    # Tier 8 — Privacy & Payments
    "XMR": "XMR-USD", "ZEC": "ZEC-USD", "DASH": "DASH-USD", "DCR": "DCR-USD",
    "SCRT": "SCRT-USD", "FLUX": "FLUX-USD", "KAS": "KAS-USD",
    # Tier 9 — Exchange tokens
    "CRO": "CRO-USD", "OKB": "OKB-USD", "LEO": "LEO-USD", "GT": "GT-USD",
    "KCS": "KCS-USD", "MX": "MX-USD",
    # Tier 10 — Oracles & Data
    "BAND": "BAND-USD", "API3": "API3-USD", "DIA": "DIA-USD", "UMA": "UMA-USD",
    "TRB": "TRB-USD", "PYTH": "PYTH-USD", "WLD": "WLD-USD", "JUP": "JUP-USD",
    "W": "W-USD", "ENA": "ENA-USD", "ETHFI": "ETHFI-USD",
    # Tier 11 — RWA & Stablecoins-adjacent
    "ONDO": "ONDO-USD", "MPL": "MPL-USD", "CFG": "CFG-USD", "POLYX": "POLYX-USD",
    "RSR": "RSR-USD", "FXS": "FXS-USD",
    # Tier 12 — More trending
    "PEOPLE": "PEOPLE-USD", "BOME": "BOME-USD", "MEW": "MEW-USD",
    "DOGS": "DOGS-USD", "NOT": "NOT-USD", "TON": "TON-USD",
    "AERO": "AERO-USD", "ZRO": "ZRO-USD", "LISTA": "LISTA-USD",
    "BB": "BB-USD", "IO": "IO-USD", "OMNI": "OMNI-USD",
    "REZ": "REZ-USD", "SAGA": "SAGA-USD", "TNSR": "TNSR-USD", "DYM": "DYM-USD",
}


def get_stock_count():
    return len(ALL_STOCKS)

def get_crypto_count():
    return len(CRYPTO_TICKERS)
