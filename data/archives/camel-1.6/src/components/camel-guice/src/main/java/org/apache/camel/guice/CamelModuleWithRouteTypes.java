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
package org.apache.camel.guice;

import java.util.Set;

import com.google.common.collect.Sets;
import com.google.inject.Injector;
import com.google.inject.Provides;
import org.apache.camel.Routes;



/**
 * A Guice Module which injects the CamelContext with the specified {@link Routes} types - which are then injected by Guice.
 * <p>
 * If you wish to bind all of the bound {@link Routes} implementations available - maybe with some filter applied - then
 * please use the {@link org.apache.camel.guice.CamelModuleWithMatchingRoutes}.
 * <p>
 * Or if you would like to specify exactly which {@link Routes} to bind then use the {@link CamelModule} and create a provider
 * method annotated with @Provides and returning Set<Routes> such as
 * <code><pre>
 * public class MyModule extends CamelModule {
 *   &#64;Provides
 *   Set&lt;Routes&gt; routes(Injector injector) { ... }
 * }
 * </pre></code>
 * 
 *
 * @version $Revision$
 */
public class CamelModuleWithRouteTypes extends CamelModule {
    private Set<Class<? extends Routes>> routes;

    public CamelModuleWithRouteTypes(Class<? extends Routes>... routes) {
        this(Sets.newHashSet(routes));
    }

    public CamelModuleWithRouteTypes(Set<Class<? extends Routes>> routes) {
        this.routes = routes;
    }

    @Provides
    Set<Routes> routes(Injector injector) {
        Set<Routes> answer = Sets.newHashSet();
        for (Class<? extends Routes> type : routes) {
            Routes route = injector.getInstance(type);
            answer.add(route);
        }
        return answer;
    }
}
