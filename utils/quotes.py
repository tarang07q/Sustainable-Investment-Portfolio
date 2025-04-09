import random

# List of finance and sustainable investing quotes
FINANCE_QUOTES = [
    {
        "quote": "The best investment you can make is in yourself.",
        "author": "Warren Buffett"
    },
    {
        "quote": "The stock market is a device for transferring money from the impatient to the patient.",
        "author": "Warren Buffett"
    },
    {
        "quote": "In the short run, the market is a voting machine, but in the long run, it is a weighing machine.",
        "author": "Benjamin Graham"
    },
    {
        "quote": "The individual investor should act consistently as an investor and not as a speculator.",
        "author": "Benjamin Graham"
    },
    {
        "quote": "The four most dangerous words in investing are: 'this time it's different.'",
        "author": "Sir John Templeton"
    },
    {
        "quote": "The goal of the investor is to find situations where it is safe not to diversify.",
        "author": "Charlie Munger"
    },
    {
        "quote": "Risk comes from not knowing what you're doing.",
        "author": "Warren Buffett"
    },
    {
        "quote": "The biggest risk of all is not taking one.",
        "author": "Mellody Hobson"
    },
    {
        "quote": "The time of maximum pessimism is the best time to buy, and the time of maximum optimism is the best time to sell.",
        "author": "Sir John Templeton"
    },
    {
        "quote": "It's not how much money you make, but how much money you keep, how hard it works for you, and how many generations you keep it for.",
        "author": "Robert Kiyosaki"
    }
]

SUSTAINABLE_INVESTING_QUOTES = [
    {
        "quote": "Sustainable investing is not just about doing good; it's about doing well by doing good.",
        "author": "Al Gore"
    },
    {
        "quote": "The future is green energy, sustainability, renewable energy.",
        "author": "Arnold Schwarzenegger"
    },
    {
        "quote": "Sustainability is no longer about doing less harm. It's about doing more good.",
        "author": "Jochen Zeitz"
    },
    {
        "quote": "We don't have to sacrifice a strong economy for a healthy environment.",
        "author": "Dennis Weaver"
    },
    {
        "quote": "Cheap fashion is really far from that. It may be cheap in terms of the financial cost, but very expensive when it comes to the environment and the cost of human life.",
        "author": "Livia Firth"
    },
    {
        "quote": "The most sustainable way to create long-term shareholder value is to continuously invest in our employees, our customers and the communities we serve.",
        "author": "Larry Fink, BlackRock"
    },
    {
        "quote": "One of the best tools we have is to show that doing the right thing for the planet can be profitable.",
        "author": "Inger Anderson, UNEP"
    },
    {
        "quote": "The insurance industry and wider financial sector have the power and responsibility to drive progress towards a net-zero economy.",
        "author": "UN Environment Programme Finance Initiative"
    },
    {
        "quote": "ESG is not about political or personal beliefs. It's about identifying and incorporating risks and opportunities into investment decisions.",
        "author": "Brian Moynihan, Bank of America"
    },
    {
        "quote": "Sustainable development is development that meets the needs of the present without compromising the ability of future generations to meet their own needs.",
        "author": "Brundtland Commission"
    }
]

# Get a random finance quote
def get_random_finance_quote():
    return random.choice(FINANCE_QUOTES)

# Get a random sustainable investing quote
def get_random_sustainable_quote():
    return random.choice(SUSTAINABLE_INVESTING_QUOTES)

# Get a random quote (from either category)
def get_random_quote():
    all_quotes = FINANCE_QUOTES + SUSTAINABLE_INVESTING_QUOTES
    return random.choice(all_quotes)
