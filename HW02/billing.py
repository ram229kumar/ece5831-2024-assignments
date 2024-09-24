def calculate_tax(price, tax):
    return price * tax


def print_billing_doc():
    tax_rate = 0.1
    products = [{'name': 'Book',  'price': 30},
                {'name': 'Pen', 'price': 5}]

    print(f'Name\tPrice\tTax')

    for product in products:
        tax = calculate_tax(product['price'], tax_rate)
        print(f"{product['name']}\t{product['price']}\t{tax}")


print(__name__)