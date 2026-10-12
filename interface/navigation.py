"""Page visibility shared by the sidebar and routing guard."""


def available_pages(is_admin):
    if is_admin:
        return ("Admin test", "Trade view", "Agent", "About", "Kelly Sizing",
                "Batch", "Market Data", "Structure Selection", "Scenario Weightings", "Query log")
    return ("Agent", "About")


def allowed_page(page, is_admin):
    return page if page in available_pages(is_admin) else "Agent"
