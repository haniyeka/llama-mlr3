jar: clean-jar
	cd java/ && javac shapleyComputation/*.java && jar cf ShapleyValue.jar shapleyComputation/ && mv ShapleyValue.jar ../inst/java/ShapleyValue.jar

clean-jar:
	rm -f java/shapleyComputation/*.class inst/java/ShapleyValue.jar
