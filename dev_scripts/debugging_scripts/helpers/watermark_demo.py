from aa_utilities.helpers import watermark

output_str = watermark(
    author="Amin Allahyar",
    email='Amin.Allahyar@astrazeneca.com',
    timezone='Europe/Stockholm',
    namespace=globals(),
)

print(output_str)
