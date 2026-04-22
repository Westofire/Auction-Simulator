"""
auction_engine.py
-----------------
Core auction mechanics for first-price and second-price (Vickrey) auctions.

Input:  bids  -> dict {bidder_id: bid_amount}
Output: (winner_id, price_paid) tuple
"""


def first_price_auction(bids: dict) -> tuple:
    """
    First-Price Sealed-Bid Auction:
    - The highest bidder wins.
    - The winner pays exactly their own bid.

    Args:
        bids (dict): {bidder_id: bid_amount}

    Returns:
        tuple: (winner_id, price_paid)

    Raises:
        ValueError: If bids dictionary is empty.
    """
    if not bids:
        raise ValueError("Bids dictionary cannot be empty.")

    winner_id = max(bids, key=lambda bidder: bids[bidder])
    price_paid = bids[winner_id]

    return winner_id, price_paid


def second_price_auction(bids: dict) -> tuple:
    """
    Second-Price Sealed-Bid Auction (Vickrey Auction):
    - The highest bidder wins.
    - The winner pays the second-highest bid (not their own).
    - If only one bidder exists, the winner pays their own bid.

    Args:
        bids (dict): {bidder_id: bid_amount}

    Returns:
        tuple: (winner_id, price_paid)

    Raises:
        ValueError: If bids dictionary is empty.
    """
    if not bids:
        raise ValueError("Bids dictionary cannot be empty.")

    sorted_bids = sorted(bids.values(), reverse=True)

    winner_id = max(bids, key=lambda bidder: bids[bidder])

    # If only one bidder, they pay their own bid; otherwise pay second-highest
    if len(sorted_bids) == 1:
        price_paid = sorted_bids[0]
    else:
        price_paid = sorted_bids[1]

    return winner_id, price_paid
