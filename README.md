# User Defined Function template for AsterixDB
This is a minimum User Defined Function (UDF) Template for AsterixDB.

### Prerequisite
* Successfully build and install (via `mvn install`) AsterixDB locally.
 (See [AsterixDB Dependecies](#asterixdb-dependencies)
for details)
* Download the [Standalone Cluster Installer](http://asterixdb.apache.org/download.html)
and go through the instructions on [AsterixDB website](http://asterixdb.apache.org/docs/0.9.0/install.html).

### How To Use
* Clone this repo onto your local machine.
* Build this project(use `mvn install` or `mvn package`).
* The UDF package will be under `target/` directory.

### AsterixDB Dependencies
The dependency to AsterixDB is specified in pom.xml as follow:
```
<dependency>
    <groupId>org.apache.asterix</groupId>
    <artifactId>asterix-external-data</artifactId>
    <version>${asterixdb-verson}</version>
    <scope>provided</scope>
</dependency>
```
There are two ways to enable Maven to pick up AsterixDB depencies: 
  * Local Maven directory (Recommended): In this case, we will need to build AsterixDB locally
  and use `mvn install` to add all
  AsterixDB libraries into the local maven directory so it can be used by this project.
  * Remote Maven Repo (Not Recommended): You can also try to add AsterixDB dependencies from remote
  Maven repo. Since
  remote repo is not always up-to-date, it is recommended to use the local maven repo.
  If you want to use remote one, you will need to change the `asterixdb-verson` in pom.xml
  to match the latest version on remote repo.

### Sample UDFs
* Sum of two numbers: 
  * This example migrated from AsterxDB calculates the sum of two given parameters.
  It reads in two integers and returns their sum as result.
* Simple sentiment analysis
  * This function appends an extra field `Sentiment` to the input record. A sample AQL to
  test this function is as follow:
 ```
 use dataverse test;

create type Tweet as open {
	"id" : string,
	"text": string
}

create type AnnotatedTweet as open {
	"id": string,
	"text": string,
	"Sentiment": string
}

let $t := {"id":"1", "text":"123123"}
return testlib#getSentiment($t)
```

* Text Encryption
  * This example shows how to use other libraries with UDF. It takes two string parameters, 
  text to be encrypted and encryption key, and it returns with the encrypted text. To run this
  example, you will need to configure the dependencies for UDF correctly. We will elaborate more
  on this in next section.

### Dependencies from UDFs
AsterxDB allows UDF to use external libraries. There are two options to resolve the dependencies of the
external libraries.
* **Option 1**: Single jar with dependencies
  * You can include all the dependencies needed into your UDF jar, i.e. create a fat jar with
  all dependencies. In this way, when you install the UDF package, all dependencies will also
  be installed on AsterixDB. The down side of this method is,
  if the size of exteral library is big, this
  will cause the booting time becomes extremely long. Thus this is not recommended if you have many/large
  external libraries. To compile a fat jar, you need to add following configuration into the pom.xml, before
  the <execution> section.
  ```
  ...
  <execution>
    <id>jar-with-dependencies</id>
    <phase>package</phase>
    <goals>
        <goal>single</goal>
    </goals>
    <configuration>
        <descriptorRefs>
            <descriptorRef>jar-with-dependencies</descriptorRef>
        </descriptorRefs>
    </configuration>
  </execution>
  <execution>
  ...
  ```
* **Option 2**: Add dependencies to AsterixDB repo
  * This method utilizes the class loading mechanism of AsterixDB. You will need to unzip
  the `asterix-server-0.9.0-binary-assembly.zip`
  under `asterix/`, and drop your library jars into `asterix-server-0.9.0-binary-assembly/repo`.
  Repack this directory into zip, and replace
  the old one under `asterix/`. **Note**, make sure the directory structure of
  the zip file is not nested. You should see following
  structure when executing `unzip -l asterix-server-0.9.0-binary-assembly.zip`.
  
  ```
  Archive:  asterix-server-0.9.0-binary-assembly.zip
  Length      Date    Time    Name
  
        0  01-19-2017 14:15   bin/
     3765  01-18-2017 19:05   bin/asterixcc
     3327  01-18-2017 19:05   bin/asterixcc.bat
     4107  01-18-2017 19:04   bin/asterixhelper
     3331  01-18-2017 19:04   bin/asterixhelper.bat
     3765  01-18-2017 19:05   bin/asterixnc
     ....
  ```
  
  
  
  
  
