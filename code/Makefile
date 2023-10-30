jar: clean-jar
	cd java/ && javac -target 1.4 -source 1.4 shapleyComputation/*.java && jar cf ShapleyValue.jar shapleyComputation/ && mv ShapleyValue.jar ../inst/java/ShapleyValue.jar

clean-jar:
	rm -f java/shapleyComputation/*.class inst/java/ShapleyValue.jar
