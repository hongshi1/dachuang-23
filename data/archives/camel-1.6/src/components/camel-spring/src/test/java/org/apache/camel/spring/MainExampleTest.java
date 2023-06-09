/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.spring;

import junit.framework.TestCase;
import org.apache.camel.spring.util.SimpleRouteBuilder;

/**
 * @version $Revision$
 */
public class MainExampleTest extends TestCase {
    public void testMain() throws Exception {
        // lets make a simple route
        SimpleRouteBuilder builder = new SimpleRouteBuilder();
        builder.setFromUri("file://src/test/data?noop=true");
        builder.setBeanClass("org.apache.camel.spring.example.MyProcessor");
        builder.setToUri("file://target/mainTest");

        Main main = new Main();
        main.addRouteBuilder(builder);
        main.start();

        // then some time later
        Thread.sleep(3000);
        main.stop();

    }
}