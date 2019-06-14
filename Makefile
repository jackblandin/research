build_release:
	python setup.py sdist

test_publish_release:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/research-${VERSION}.tar.gz

publish_release:
	twine upload dist/research-${VERSION}.tar.gz
